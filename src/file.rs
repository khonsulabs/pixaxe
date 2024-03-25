use std::borrow::{Borrow, BorrowMut, Cow};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::num::TryFromIntError;
use std::ops::{Deref, DerefMut, Index, Sub};
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::{fs, io};

use cushy::figures::units::UPx;
use cushy::figures::Size;
use cushy::styles::Color;
use kempt::{map, Map};
use ordered_varint::Variable;
use q_compress::DEFAULT_COMPRESSION_LEVEL;

use crate::{
    BlendMode, Edit, EditOp, FileData, Image, ImageChanges, ImageFile, Layer, LayerChange,
    LayerDelta, LayerId, PaletteChange,
};

#[derive(Debug)]
pub struct File {
    path: PathBuf,
    last_edit: Option<SystemTime>,
    on_disk: fs::File,
    layers: Map<LayerId, Layer>,
}

impl File {
    pub fn create(path: PathBuf, data: &mut FileData) -> io::Result<Self> {
        let mut on_disk = fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&path)?;

        let mut writer = CrcBufWriter::new(&mut on_disk)?;

        Chunk::Version(0).write_to(&mut writer)?;
        let mut layers = Map::new();
        let mut last_edit = None;
        Self::save_inner(writer, &mut layers, &mut last_edit, data)?;

        Ok(Self {
            path,
            last_edit,
            on_disk,
            layers,
        })
    }

    pub fn load(path: PathBuf) -> io::Result<ImageFile> {
        let mut file = fs::OpenOptions::new().read(true).write(true).open(&path)?;
        let length = file.seek(SeekFrom::End(0))?;
        file.seek(SeekFrom::Start(0))?;
        let mut reader = BufReader::new(&mut file);

        let mut scratch = ScratchBuffer::default();
        match Chunk::read_from(&mut reader, FilePos(0), &mut scratch)? {
            Chunk::Version(0) => {}
            Chunk::Version(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unsupported file version",
                ))
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "incorrect first chunk",
                ))
            }
        }

        let header = Header::load_latest(&mut reader, &mut scratch, length)?;
        let mut history = VecDeque::new();

        let mut edit_pos = header.last_edit;
        let mut last_edit = None;
        while edit_pos.0 > 0 {
            reader.seek(SeekFrom::Start(edit_pos.0))?;
            let Chunk::Edit(entry) = Chunk::read_from(&mut reader, edit_pos, &mut scratch)? else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "expected edit chunk",
                ));
            };
            last_edit =
                Some(last_edit.map_or(entry.edit.when, |last_edit| entry.edit.when.max(last_edit)));
            history.push_front(entry.edit.into_owned());
            edit_pos = entry.previous;
        }

        let mut layers = Vec::new();
        let mut file_layers = Map::new();
        for layer_pos in &*header.layers {
            reader.seek(SeekFrom::Start(layer_pos.0))?;
            let Chunk::Layer(layer) = Chunk::read_from(&mut reader, *layer_pos, &mut scratch)?
            else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "expected layer chunk",
                ));
            };
            let layer = layer.into_owned();
            file_layers.insert(layer.id, layer.clone());
            layers.push(layer);
        }
        drop(reader);

        let file = File {
            path,
            last_edit,
            on_disk: file,
            layers: file_layers,
        };

        Ok(ImageFile {
            data: FileData {
                image: Image {
                    size: header.size,
                    layers,
                    palette: header.palette.into_owned(),
                },
                history: Vec::from(history),
                undone: Vec::new(),
            },
            on_disk: Some(file),
        })
    }

    pub fn save_as(&mut self, new_path: PathBuf, data: &mut FileData) -> io::Result<()> {
        let mut new_file = fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&new_path)?;
        self.on_disk.seek(SeekFrom::Start(0))?;
        io::copy(&mut self.on_disk, &mut new_file)?;
        self.on_disk = new_file;
        self.path = new_path;

        self.save(data)
    }

    pub fn save(&mut self, data: &mut FileData) -> io::Result<()> {
        Self::save_inner(
            CrcBufWriter::new(&mut self.on_disk)?,
            &mut self.layers,
            &mut self.last_edit,
            data,
        )?;

        Ok(())
    }

    fn save_inner(
        mut writer: CrcBufWriter<'_>,
        disk_layers: &mut Map<LayerId, Layer>,
        last_edit: &mut Option<SystemTime>,
        data: &mut FileData,
    ) -> io::Result<()> {
        let current_last = data.history.last().map(|c| c.when);
        if last_edit == &current_last {
            // No changes.
            return Ok(());
        } else {
            *last_edit = current_last;
        }
        let mut previous = FilePos::default();
        for edit in &mut data.history {
            if edit.file_offset.0 == 0 {
                edit.file_offset = Chunk::Edit(EditEntry {
                    edit: Exclusive::Ref(edit),
                    previous,
                })
                .write_to(&mut writer)?;
            }
            previous = edit.file_offset;
        }

        let mut layer_positions = Vec::with_capacity(data.image.layers.len());
        for layer in &mut data.image.layers {
            let disk_layer = disk_layers.entry(layer.id);
            let changed = match &disk_layer {
                map::Entry::Occupied(disk_layer) => &**disk_layer != layer,
                map::Entry::Vacant(_) => true,
            };
            if changed || layer.file_offset.0 == 0 {
                let offset = Chunk::Layer(Exclusive::Ref(layer)).write_to(&mut writer)?;
                layer.file_offset = offset;
                disk_layer
                    .and_modify(|disk_layer| disk_layer.clone_from(layer))
                    .or_insert_with(|| layer.clone());
            }
            layer_positions.push(layer.file_offset);
        }

        Chunk::Header(Header {
            last_edit: data
                .history
                .last()
                .map_or_else(FilePos::default, |edit| edit.file_offset),
            size: data.image.size,
            layers: Cow::Borrowed(&layer_positions),
            palette: Cow::Borrowed(&data.image.palette),
        })
        .write_to(&mut writer)?;

        let file = writer.finish()?;
        file.sync_all()?;
        Ok(())
    }
}

// Chunked file with trailers
// paxe <type> <data> <length>
// Beginning of file is always a

trait Writeable {
    fn write_to(&mut self, file: &mut CrcBufWriter<'_>) -> io::Result<()>;

    #[allow(unused_variables)]
    fn write_dependencies(&mut self, file: &mut CrcBufWriter<'_>) -> io::Result<()> {
        Ok(())
    }
}

trait Readable: Sized {
    fn read_from<R: Read>(
        reader: R,
        position: FilePos,
        scratch: &mut ScratchBuffer,
    ) -> io::Result<Self>;
}

impl Writeable for Layer {
    fn write_to(&mut self, file: &mut CrcBufWriter<'_>) -> io::Result<()> {
        self.id.0.encode_variable(&mut *file)?;
        file.write_all(&[self.blend as u8])?;
        let compressed = q_compress::auto_compress(&self.data, DEFAULT_COMPRESSION_LEVEL);
        compressed.len().encode_variable(&mut *file)?;
        file.write_all(&compressed)
    }
}

impl Readable for Layer {
    fn read_from<R: Read>(
        mut reader: R,
        position: FilePos,
        scratch: &mut ScratchBuffer,
    ) -> io::Result<Self> {
        let id = LayerId(u64::decode_variable(&mut reader)?);
        reader.read_exact(scratch.slice_mut(1))?;
        let blend = BlendMode::try_from(scratch[0])?;
        let data_len = usize::decode_variable(&mut reader)?;
        scratch.resize(data_len);
        reader.read_exact(scratch.slice_mut(data_len))?;
        let data = q_compress::auto_decompress(scratch.slice(data_len))
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(Self {
            id,
            data,
            blend,
            file_offset: position,
        })
    }
}

impl Writeable for Edit {
    fn write_to(&mut self, file: &mut CrcBufWriter<'_>) -> io::Result<()> {
        let ts = self
            .when
            .duration_since(UNIX_EPOCH)
            .expect("invalid timestamp")
            .as_secs();
        ts.encode_variable(&mut *file)?;
        file.write_all(&[self.op as u8])?;
        self.changes.layers.len().encode_variable(&mut *file)?;
        for layer in &mut self.changes.layers {
            layer.write_to(file)?;
        }
        self.changes.palette.len().encode_variable(&mut *file)?;
        for palette in &mut self.changes.palette {
            palette.write_to(file)?;
        }

        Ok(())
    }

    fn write_dependencies(&mut self, file: &mut CrcBufWriter<'_>) -> io::Result<()> {
        for layer in &mut self.changes.layers {
            layer.write_dependencies(file)?;
        }
        Ok(())
    }
}

impl Readable for Edit {
    fn read_from<R: Read>(
        mut reader: R,
        position: FilePos,
        scratch: &mut ScratchBuffer,
    ) -> io::Result<Self> {
        let when = u64::decode_variable(&mut reader)?;
        reader.read_exact(scratch.slice_mut(1))?;
        let op = EditOp::try_from(scratch[0])?;
        let mut changes = ImageChanges::default();
        let layer_changes = usize::decode_variable(&mut reader)?;
        for _ in 0..layer_changes {
            changes
                .layers
                .push(LayerChange::read_from(&mut reader, position, scratch)?);
        }
        let palette_changes = usize::decode_variable(&mut reader)?;
        for _ in 0..palette_changes {
            changes.palette.push(dbg!(PaletteChange::read_from(
                &mut reader,
                position,
                scratch
            ))?);
        }

        Ok(Self {
            when: UNIX_EPOCH + Duration::from_secs(when),
            op,
            changes,
            file_offset: position,
        })
    }
}

impl Writeable for LayerChange {
    fn write_to(&mut self, file: &mut CrcBufWriter<'_>) -> io::Result<()> {
        let (kind, index, layer) = match self {
            LayerChange::Insert(index, layer) => (ChangeKind::Insert, index, layer),
            LayerChange::Remove(index, layer) => (ChangeKind::Remove, index, layer),
            LayerChange::Change(index, delta) => {
                file.write_all(&[ChangeKind::Change as u8])?;
                index.encode_variable(&mut *file)?;
                return delta.write_to(&mut *file);
            }
        };

        file.write_all(&[kind as u8])?;
        index.encode_variable(&mut *file)?;
        layer.file_offset.write_to(&mut *file)?;

        Ok(())
    }

    fn write_dependencies(&mut self, file: &mut CrcBufWriter<'_>) -> io::Result<()> {
        match self {
            LayerChange::Remove(_, layer) | LayerChange::Insert(_, layer) => {
                if layer.file_offset.0 == 0 {
                    layer.file_offset = Chunk::Layer(Exclusive::Ref(layer)).write_to(file)?;
                }
            }
            LayerChange::Change(_, _) => {}
        }
        Ok(())
    }
}

impl Readable for LayerChange {
    fn read_from<R: Read>(
        mut reader: R,
        position: FilePos,
        scratch: &mut ScratchBuffer,
    ) -> io::Result<Self> {
        reader.read_exact(scratch.slice_mut(1))?;
        let kind = ChangeKind::try_from(scratch[0])?;
        let index = usize::decode_variable(&mut reader)?;
        match kind {
            ChangeKind::Insert => {
                let position = FilePos::read_from(&mut reader)?;
                Ok(LayerChange::Insert(
                    index,
                    Layer::read_from(reader, position, scratch)?,
                ))
            }
            ChangeKind::Remove => {
                let position = FilePos::read_from(&mut reader)?;
                Ok(LayerChange::Insert(
                    index,
                    Layer::read_from(reader, position, scratch)?,
                ))
            }
            ChangeKind::Change => Ok(LayerChange::Change(
                index,
                LayerDelta::read_from(reader, position, scratch)?,
            )),
        }
    }
}

impl Writeable for LayerDelta {
    fn write_to(&mut self, file: &mut CrcBufWriter<'_>) -> io::Result<()> {
        let compressed = q_compress::auto_compress(&self.0, DEFAULT_COMPRESSION_LEVEL);
        compressed.len().encode_variable(&mut *file)?;
        file.write_all(&compressed)
    }
}

impl Readable for LayerDelta {
    fn read_from<R: Read>(
        mut reader: R,
        _position: FilePos,
        scratch: &mut ScratchBuffer,
    ) -> io::Result<Self> {
        let data_len = usize::decode_variable(&mut reader)?;
        scratch.resize(data_len);
        reader.read_exact(scratch.slice_mut(data_len))?;
        let data = q_compress::auto_decompress(scratch.slice(data_len))
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(Self(data))
    }
}
impl Writeable for PaletteChange {
    fn write_to(&mut self, file: &mut CrcBufWriter<'_>) -> io::Result<()> {
        let (kind, index, color) = match self {
            PaletteChange::InsertColor(index, color) => (ChangeKind::Insert, index, color),
            PaletteChange::RemoveColor(index, color) => (ChangeKind::Remove, index, color),
            PaletteChange::Change(index, color) => (ChangeKind::Change, index, color),
        };

        file.write_all(&[kind as u8])?;
        index.encode_variable(&mut *file)?;
        file.write_all(&color.0.to_le_bytes())?;

        Ok(())
    }
}

impl Readable for PaletteChange {
    fn read_from<R: Read>(
        mut reader: R,
        _position: FilePos,
        scratch: &mut ScratchBuffer,
    ) -> io::Result<Self> {
        reader.read_exact(scratch.slice_mut(1))?;
        let kind = ChangeKind::try_from(scratch[0])?;
        let index = usize::decode_variable(&mut reader)?;
        let mut color = [0; 4];
        reader.read_exact(&mut color)?;
        let color = Color(u32::from_le_bytes(color));
        match kind {
            ChangeKind::Insert => Ok(PaletteChange::InsertColor(index, color)),
            ChangeKind::Remove => Ok(PaletteChange::RemoveColor(index, color)),
            ChangeKind::Change => Ok(PaletteChange::Change(index, color)),
        }
    }
}

enum ChangeKind {
    Insert = 0,
    Remove,
    Change,
}

impl TryFrom<u8> for ChangeKind {
    type Error = InvalidChangeKind;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Insert),
            1 => Ok(Self::Remove),
            2 => Ok(Self::Change),
            _ => Err(InvalidChangeKind),
        }
    }
}

pub struct InvalidChangeKind;

impl From<InvalidChangeKind> for io::Error {
    fn from(_value: InvalidChangeKind) -> Self {
        io::Error::new(io::ErrorKind::InvalidData, "invalid change kind")
    }
}

enum ChunkKind {
    Version = 0,
    Header,
    Layer,
    Edit,
}

impl TryFrom<u8> for ChunkKind {
    type Error = InvalidChunkKind;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Version),
            1 => Ok(Self::Header),
            2 => Ok(Self::Layer),
            3 => Ok(Self::Edit),
            _ => Err(InvalidChunkKind),
        }
    }
}

pub struct InvalidChunkKind;

impl From<InvalidChunkKind> for io::Error {
    fn from(_value: InvalidChunkKind) -> Self {
        io::Error::new(io::ErrorKind::InvalidData, "invalid chunk kind")
    }
}

struct Tail {
    crc: u32,
    length: u32,
}

impl Tail {
    fn read_from(mut reader: impl Read) -> io::Result<Self> {
        let mut bytes = [0; 4];
        reader.read_exact(&mut bytes)?;
        let crc = u32::from_le_bytes(bytes);
        reader.read_exact(&mut bytes)?;
        let length = u32::from_le_bytes(bytes);
        Ok(Self { crc, length })
    }

    fn check(&self, crc: u32, length: usize) -> io::Result<()> {
        if checked_cast::<_, usize>(self.length)? != length {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid data length",
            ));
        }

        if crc != self.crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid chunk tail",
            ));
        }

        Ok(())
    }
}

enum Exclusive<'a, T>
where
    T: ToOwned + ?Sized,
{
    Ref(&'a mut T),
    Owned(T::Owned),
}

impl<T> Exclusive<'_, T>
where
    T: ToOwned + ?Sized,
{
    fn into_owned(self) -> T::Owned {
        match self {
            Exclusive::Ref(value) => value.to_owned(),
            Exclusive::Owned(value) => value,
        }
    }
}

impl<T> Deref for Exclusive<'_, T>
where
    T: ToOwned + ?Sized,
    T::Owned: Borrow<T>,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Exclusive::Ref(value) => value,
            Exclusive::Owned(value) => value.borrow(),
        }
    }
}

impl<T> DerefMut for Exclusive<'_, T>
where
    T: ToOwned + ?Sized,
    T::Owned: BorrowMut<T>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Exclusive::Ref(value) => value,
            Exclusive::Owned(value) => value.borrow_mut(),
        }
    }
}

enum Chunk<'a> {
    Version(u8),
    Header(Header<'a>),
    Layer(Exclusive<'a, Layer>),
    Edit(EditEntry<'a>),
}

impl Chunk<'_> {
    const OPENER: &'static [u8; 4] = b"paxe";
    const TAIL_LEN: usize = 12;
    const TRAILER: &'static [u8; 4] = b"Paxe";

    fn write_to(&mut self, file: &mut CrcBufWriter<'_>) -> io::Result<FilePos> {
        match self {
            Chunk::Version(_) => {}
            Chunk::Header(header) => header.write_dependencies(file)?,
            Chunk::Layer(layer) => layer.write_dependencies(file)?,
            Chunk::Edit(edit) => edit.write_dependencies(file)?,
        }
        let start = file.position();
        file.write_all(Self::OPENER)?;
        match self {
            Chunk::Version(version) => {
                file.write_all(&[ChunkKind::Version as u8])?;
                file.enable_crc();
                file.write_all(&[*version])?;
            }
            Chunk::Header(header) => {
                file.write_all(&[ChunkKind::Header as u8])?;
                file.enable_crc();
                header.write_to(file)?;
            }
            Chunk::Layer(layer) => {
                file.write_all(&[ChunkKind::Layer as u8])?;
                file.enable_crc();
                layer.write_to(file)?;
            }
            Chunk::Edit(entry) => {
                file.write_all(&[ChunkKind::Edit as u8])?;
                file.enable_crc();
                entry.write_to(file)?;
            }
        }

        let end = file.position();

        let crc = file.take_crc();
        file.write_all(&crc.to_le_bytes())?;

        // we subtract 5 from the length, 4 for the opener and 1 for the
        // ChunkKind.
        let length = u32::try_from(end - start - 5)
            .map_err(|err| io::Error::new(io::ErrorKind::OutOfMemory, err))?;
        file.write_all(&length.to_le_bytes())?;
        file.write_all(Self::TRAILER)?;

        Ok(start)
    }

    fn check_tail(crc: u32, length: usize, tail: &[u8]) -> io::Result<()> {
        if tail.len() != Self::TAIL_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid chunk tail length",
            ));
        } else if tail[8..12] != Self::TRAILER[..] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid chunk trailer",
            ));
        }
        let tail = Tail::read_from(&tail[0..8])?;
        tail.check(crc, length)
    }
}

impl Readable for Chunk<'static> {
    fn read_from<R: Read>(
        mut reader: R,
        position: FilePos,
        scratch: &mut ScratchBuffer,
    ) -> io::Result<Self> {
        scratch.resize(Self::TAIL_LEN);

        reader.read_exact(scratch.slice_mut(5))?;
        let mut reader = CrcReader::new(reader);

        if scratch.slice(4) != &Self::OPENER[..] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "missing magic code",
            ));
        }
        let chunk = match scratch[4] {
            0 => {
                reader.read_exact(scratch.slice_mut(1))?;
                Chunk::Version(scratch[0])
            }
            1 => Chunk::Header(Header::read_from(&mut reader, position, scratch)?),
            2 => Chunk::Layer(Exclusive::Owned(Layer::read_from(
                &mut reader,
                position,
                scratch,
            )?)),
            3 => Chunk::Edit(EditEntry::read_from(&mut reader, position, scratch)?),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unknown chunk kind",
                ))
            }
        };
        let (mut reader, crc, bytes_read) = reader.finish();
        reader.read_exact(scratch.slice_mut(Self::TAIL_LEN))?;
        Self::check_tail(crc, bytes_read, scratch.slice(Self::TAIL_LEN))?;
        Ok(chunk)
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug, Default)]
pub struct FilePos(u64);

impl Sub for FilePos {
    type Output = u64;

    fn sub(self, rhs: Self) -> Self::Output {
        self.0 - rhs.0
    }
}

impl FilePos {
    fn write_to(&self, file: &mut CrcBufWriter<'_>) -> io::Result<()> {
        self.0.encode_variable(file)?;
        Ok(())
    }
}

impl FilePos {
    fn read_from<R: Read>(reader: R) -> io::Result<Self> {
        u64::decode_variable(reader).map(Self)
    }
}

#[derive(Clone, Debug)]
struct Header<'a> {
    last_edit: FilePos,
    size: Size<UPx>,
    layers: Cow<'a, [FilePos]>,
    palette: Cow<'a, [Color]>,
}

impl Header<'static> {
    fn load_latest(
        reader: &mut BufReader<&mut fs::File>,
        scratch: &mut ScratchBuffer,
        file_length: u64,
    ) -> io::Result<Self> {
        if file_length < 4 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too short"));
        }
        let buffer_size = checked_cast::<_, u64>(reader.capacity())?;
        let mut scan_from = file_length.saturating_sub(buffer_size);

        // Optimal case: the tail of the header is at the end of the file.
        if reader.buffer().ends_with(Chunk::TRAILER) {
            if let Ok(result) = Self::read_from_tail(reader, scratch, file_length) {
                return Ok(result);
            }
        }

        // When not, we need to scan. We're going to utilize the bufreader's
        // buffer, but we also have to worry about the magic code being split
        // across bufferings. Since the magic code is 4 bytes, the maximum
        // remaining after a split is 3 bytes.
        let mut last_three_bytes = [0, 0, 0];
        while !reader.buffer().is_empty() {
            let buffer = reader.buffer();
            // Check for a split magic code.
            let overlapped_tail = match buffer.last().expect("non-zero") {
                b'P' if last_three_bytes == *b"axe" => Some(1),
                b'a' if last_three_bytes[0..2] == *b"xe" && buffer[buffer.len() - 2] == b'P' => {
                    Some(2)
                }
                b'x' if last_three_bytes[0] == b'e' && buffer[buffer.len() - 3..2] == *b"Pa" => {
                    Some(3)
                }
                _ => None,
            };
            // Copy the first three bytes for the next iteration. Doing this now
            // allows us to not worry about the effects of seeking while trying
            // to read headers.
            last_three_bytes.copy_from_slice(&buffer[0..3]);
            if let Some(overlapped_tail) = overlapped_tail {
                // We found a magic code that was split across the buffer.
                if let Ok(result) = Self::read_from_tail(
                    reader,
                    scratch,
                    scan_from + buffer_size - overlapped_tail + 4,
                ) {
                    return Ok(result);
                }
            }

            let mut scan_end = reader.buffer().len();
            // Reverse scan through the buffer for the trailer magic code.
            while scan_end > 0 {
                let Some(offset) =
                    memchr::memmem::rfind(&reader.buffer()[0..scan_end], Chunk::TRAILER)
                else {
                    break;
                };
                if let Ok(result) = Self::read_from_tail(
                    reader,
                    scratch,
                    scan_from + checked_cast::<_, u64>(offset)? + 4,
                ) {
                    return Ok(result);
                }
                scan_end = offset;
            }

            if scan_from == 0 {
                break;
            } else {
                // We didn't find a trailer, seek to the next location and
                // refill the buffer.
                scan_from = scan_from.saturating_sub(buffer_size);
                reader.seek(SeekFrom::Start(scan_from))?;
                reader.fill_buf()?;
            }
        }

        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "header not found",
        ))
    }

    fn read_from_tail(
        reader: &mut BufReader<&mut fs::File>,
        scratch: &mut ScratchBuffer,
        tail: u64,
    ) -> io::Result<Self> {
        reader.seek(SeekFrom::Start(tail - 12))?; // TODO TAIL LENGTH
        let mut tail_data = [0; 8];
        reader.read_exact(&mut tail_data)?;
        let tail_data = Tail::read_from(&tail_data[..])?;

        let Some(start) = tail
            .checked_sub(u64::from(tail_data.length))
            .and_then(|t| t.checked_sub(17))
        else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid length"));
        };
        reader.seek(SeekFrom::Start(start))?;
        let Chunk::Header(header) = Chunk::read_from(reader, FilePos(start), scratch)? else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "expected header chunk",
            ));
        };
        Ok(header)
    }
}

impl Writeable for Header<'_> {
    fn write_to(&mut self, file: &mut CrcBufWriter<'_>) -> io::Result<()> {
        self.last_edit.0.encode_variable(&mut *file)?;
        self.size.width.get().encode_variable(&mut *file)?;
        self.size.height.get().encode_variable(&mut *file)?;
        self.layers.len().encode_variable(&mut *file)?;
        for pos in &*self.layers {
            pos.write_to(&mut *file)?;
        }
        self.palette.len().encode_variable(&mut *file)?;
        for color in &*self.palette {
            file.write_all(&color.0.to_le_bytes())?;
        }

        Ok(())
    }
}

impl Readable for Header<'static> {
    fn read_from<R: Read>(
        mut reader: R,
        _position: FilePos,
        _scratch: &mut ScratchBuffer,
    ) -> io::Result<Self> {
        let last_edit = FilePos::read_from(&mut reader)?;
        let width = u32::decode_variable(&mut reader)?;
        let height = u32::decode_variable(&mut reader)?;
        let layer_count = usize::decode_variable(&mut reader)?;
        let mut layers = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            layers.push(FilePos::read_from(&mut reader)?);
        }

        let palette_count = usize::decode_variable(&mut reader)?;
        let mut palette = Vec::with_capacity(palette_count);
        for _ in 0..palette_count {
            let mut bytes = [0; 4];
            reader.read_exact(&mut bytes)?;
            palette.push(Color(u32::from_le_bytes(bytes)));
        }

        Ok(Self {
            last_edit,
            size: Size::new(UPx::new(width), UPx::new(height)),
            layers: Cow::Owned(layers),
            palette: Cow::Owned(palette),
        })
    }
}

struct EditEntry<'a> {
    edit: Exclusive<'a, Edit>,
    previous: FilePos,
}

impl Writeable for EditEntry<'_> {
    fn write_to(&mut self, file: &mut CrcBufWriter<'_>) -> io::Result<()> {
        self.previous.0.encode_variable(&mut *file)?;
        self.edit.write_to(file)
    }

    fn write_dependencies(&mut self, file: &mut CrcBufWriter<'_>) -> io::Result<()> {
        self.edit.write_dependencies(file)
    }
}

impl Readable for EditEntry<'static> {
    fn read_from<R: Read>(
        mut reader: R,
        position: FilePos,
        scratch: &mut ScratchBuffer,
    ) -> io::Result<Self> {
        let previous = FilePos::read_from(&mut reader)?;
        let edit = Edit::read_from(&mut reader, position, scratch)?;
        Ok(Self {
            edit: Exclusive::Owned(edit),
            previous,
        })
    }
}

fn checked_cast<From, To>(value: From) -> io::Result<To>
where
    To: TryFrom<From, Error = TryFromIntError>,
{
    To::try_from(value).map_err(|err| io::Error::new(io::ErrorKind::OutOfMemory, err))
}

struct CrcReader<R> {
    reader: R,
    crc: crc32fast::Hasher,
    total_read: usize,
}

impl<R> CrcReader<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            crc: crc32fast::Hasher::new(),
            total_read: 0,
        }
    }

    fn finish(self) -> (R, u32, usize) {
        (self.reader, self.crc.finalize(), self.total_read)
    }
}

impl<R> Read for CrcReader<R>
where
    R: Read,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let bytes_read = self.reader.read(buf)?;
        if bytes_read > 0 {
            self.crc.update(&buf[0..bytes_read]);
            self.total_read += bytes_read;
        }
        Ok(bytes_read)
    }
}

struct CrcBufWriter<'a> {
    writer: BufWriter<&'a mut fs::File>,
    position: u64,
    crc: Option<crc32fast::Hasher>,
}

impl Write for CrcBufWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let bytes_written = self.writer.write(buf)?;
        if let Some(crc) = &mut self.crc {
            crc.update(&buf[0..bytes_written]);
        }
        self.position = self
            .position
            .checked_add(checked_cast::<_, u64>(bytes_written)?)
            .ok_or_else(|| io::Error::from(io::ErrorKind::OutOfMemory))?;
        Ok(bytes_written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

impl Seek for CrcBufWriter<'_> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        assert!(self.crc.is_none());

        let new_position = self.writer.seek(pos)?;
        self.position = new_position;
        Ok(new_position)
    }
}

impl<'a> CrcBufWriter<'a> {
    fn new(file: &'a mut fs::File) -> io::Result<Self> {
        let position = file.stream_position()?;
        Ok(Self {
            writer: BufWriter::new(file),
            position,
            crc: None,
        })
    }

    const fn position(&self) -> FilePos {
        FilePos(self.position)
    }

    fn take_crc(&mut self) -> u32 {
        self.crc.take().expect("crc not enabled").finalize()
    }

    fn enable_crc(&mut self) {
        self.crc = Some(crc32fast::Hasher::new());
    }

    fn finish(self) -> io::Result<&'a mut fs::File> {
        Ok(self.writer.into_inner()?)
    }
}

#[derive(Default)]
struct ScratchBuffer(Vec<u8>);

impl ScratchBuffer {
    fn resize(&mut self, new_length: usize) {
        if self.0.len() < new_length {
            self.0.resize(new_length, 0);
        }
    }

    fn slice(&mut self, length: usize) -> &[u8] {
        &self.0[0..length]
    }

    fn slice_mut(&mut self, length: usize) -> &mut [u8] {
        &mut self.0[0..length]
    }
}

impl Index<usize> for ScratchBuffer {
    type Output = u8;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
