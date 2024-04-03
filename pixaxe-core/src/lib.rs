use std::collections::VecDeque;
use std::fmt::Display;
use std::io;
use std::num::NonZeroU16;
use std::ops::{AddAssign, Sub, SubAssign};
use std::path::PathBuf;
use std::time::SystemTime;

use cushy::figures::units::UPx;
use cushy::figures::{Point, Size};
use cushy::styles::Color;
use file::File;
use kempt::Set;
use muse::symbol::Symbol;
use q_compress::data_types::NumberLike;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

mod file;
pub use file::FilePos;

pub static BUILTIN_SCRIPT: &str = include_str!("./builtin.muse");

#[derive(Debug)]

pub struct FileData {
    pub image: Image,
    pub history: Vec<Edit>,
    pub undone: Vec<Edit>,
}

#[derive(Debug)]
pub struct ImageFile {
    pub data: FileData,
    on_disk: Option<File>,
}

impl ImageFile {
    pub fn new(image: Image) -> Self {
        Self {
            data: FileData {
                image,
                history: Vec::new(),
                undone: Vec::new(),
            },
            on_disk: None,
        }
    }

    pub fn load(path: PathBuf) -> io::Result<Self> {
        File::load(path)
    }

    pub const fn on_disk(&self) -> bool {
        self.on_disk.is_some()
    }

    pub fn save_as(&mut self, path: PathBuf) -> io::Result<()> {
        if let Some(disk_state) = &mut self.on_disk {
            disk_state.save_as(path, &mut self.data)
        } else {
            self.on_disk = Some(File::create(path, &mut self.data)?);
            Ok(())
        }
    }

    pub fn save(&mut self) -> io::Result<()> {
        let Some(disk_state) = &mut self.on_disk else {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "image path never given",
            ));
        };

        disk_state.save(&mut self.data)
    }
}

#[derive(Debug, Clone)]
pub struct Image {
    pub size: Size<UPx>,
    pub layers: Vec<Layer>,
    pub palette: Vec<Color>,
}

impl Image {
    pub fn layer_mut(&mut self, index: usize) -> ImageLayer<'_> {
        ImageLayer {
            image: self,
            layer: index,
        }
    }

    pub fn changes(&mut self, other: &Image) -> ImageChanges {
        let mut layer_changes = Vec::new();

        let mut found_layers = Set::new();
        'layers: for (index, layer) in self.layers.iter_mut().enumerate() {
            if let Some(other) = other
                .layers
                .get(index)
                .and_then(|other| (other.id == layer.id).then_some(other))
            {
                found_layers.insert(layer.id);
                if let Some(change) = layer.changes(other) {
                    layer_changes.push(LayerChange::Change(index, change));
                }
            } else {
                for other in &other.layers {
                    if layer.id == other.id {
                        found_layers.insert(layer.id);
                        if let Some(change) = layer.changes(other) {
                            layer_changes.push(LayerChange::Change(index, change));
                        }
                        continue 'layers;
                    }
                }

                // Our layer didn't exist, insert it instead
                layer_changes.push(LayerChange::Insert(index, layer.clone()));
            }
        }

        for (index, other) in other
            .layers
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, layer)| !found_layers.contains(&layer.id))
        {
            layer_changes.insert(0, LayerChange::Remove(index, other.clone()));
        }

        let mut palette_changes = Vec::new();
        let mut found_colors = Set::new();
        for (index, color) in self.palette.iter().copied().enumerate() {
            if other
                .palette
                .get(index)
                .map_or(false, |other| *other == color)
            {
                found_colors.insert(color.0);
            } else {
                'colors: for other in other.palette.iter().copied() {
                    if color == other {
                        found_colors.insert(color.0);
                        continue 'colors;
                    }
                }

                // Our layer didn't exist, insert it instead
                palette_changes.push(PaletteChange::InsertColor(index, color));
            }
        }

        for (index, other) in other
            .palette
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, color)| !found_colors.contains(&color.0))
        {
            palette_changes.insert(0, PaletteChange::RemoveColor(index, *other));
        }

        ImageChanges {
            layers: layer_changes,
            palette: palette_changes,
        }
    }

    pub fn coordinate_to_offset(&self, coord: impl ImageCoordinate) -> Option<usize> {
        let coord = coord.to_pixel()?;
        let width = self.size.width.get();
        let height = self.size.height.get();
        if coord.x >= width || coord.y >= height {
            return None;
        }

        usize::try_from(coord.x + coord.y * width).ok()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ImageChanges {
    pub palette: Vec<PaletteChange>,
    pub layers: Vec<LayerChange>,
}

impl ImageChanges {
    pub fn apply(&self, image: &mut Image) {
        for layer in &self.layers {
            match layer {
                LayerChange::Insert(index, layer) => {
                    image.layers.insert(*index, layer.clone());
                }
                LayerChange::Change(index, changes) => {
                    changes.apply(&mut image.layers[*index]);
                }
                LayerChange::Remove(index, _) => {
                    image.layers.remove(*index);
                }
            }
        }

        for palette in &self.palette {
            palette.apply(&mut image.palette);
        }
    }

    pub fn revert(&self, image: &mut Image) {
        for layer in self.layers.iter().rev() {
            match layer {
                LayerChange::Insert(index, _) => {
                    image.layers.remove(*index);
                }
                LayerChange::Change(index, changes) => {
                    changes.revert(&mut image.layers[*index]);
                }
                LayerChange::Remove(index, layer) => {
                    image.layers.insert(*index, layer.clone());
                }
            }
        }
        for palette in self.palette.iter().rev() {
            palette.revert(&mut image.palette);
        }
    }
}

impl ImageChanges {
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty() && self.palette.is_empty()
    }
}

#[derive(Default, Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Pixel(Option<PaletteColor>);

impl Pixel {
    pub const fn clear() -> Self {
        Self(None)
    }

    pub const fn indexed(index: u16) -> Self {
        assert!(index < u16::MAX);
        Self::from_u16(index + 1)
    }

    pub const fn into_u16(self) -> u16 {
        match self.0 {
            Some(index) => index.0.get(),
            None => 0,
        }
    }

    pub const fn from_u16(index: u16) -> Self {
        match NonZeroU16::new(index) {
            Some(non_zero) => Pixel(Some(PaletteColor(non_zero))),
            None => Pixel(None),
        }
    }

    pub const fn is_some(self) -> bool {
        self.0.is_some()
    }

    pub const fn index(self) -> Option<u16> {
        if let Some(index) = self.0 {
            Some(index.0.get() - 1)
        } else {
            None
        }
    }
}

impl Display for Pixel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.into_u16(), f)
    }
}

impl NumberLike for Pixel {
    type Signed = <u16 as NumberLike>::Signed;
    type Unsigned = <u16 as NumberLike>::Unsigned;

    const HEADER_BYTE: u8 = u16::HEADER_BYTE;
    const PHYSICAL_BITS: usize = u16::PHYSICAL_BITS;

    fn to_unsigned(self) -> Self::Unsigned {
        self.into_u16()
    }

    fn from_unsigned(off: Self::Unsigned) -> Self {
        Self::from_u16(u16::from_unsigned(off))
    }

    fn to_signed(self) -> Self::Signed {
        self.into_u16().to_signed()
    }

    fn from_signed(signed: Self::Signed) -> Self {
        Self::from_u16(u16::from_signed(signed))
    }

    fn to_bytes(self) -> Vec<u8> {
        self.into_u16().to_bytes()
    }

    fn from_bytes(bytes: &[u8]) -> q_compress::errors::QCompressResult<Self> {
        u16::from_bytes(bytes).map(Self::from_u16)
    }
}

impl Sub for Pixel {
    type Output = u16;

    fn sub(self, rhs: Self) -> Self::Output {
        self.into_u16().wrapping_sub(rhs.into_u16())
    }
}

impl SubAssign<u16> for Pixel {
    fn sub_assign(&mut self, rhs: u16) {
        *self = Self::from_u16(self.into_u16().wrapping_sub(rhs));
    }
}

impl AddAssign<u16> for Pixel {
    fn add_assign(&mut self, rhs: u16) {
        *self = Self::from_u16(self.into_u16().wrapping_add(rhs));
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub struct LayerId(u64);

impl LayerId {
    pub const fn first() -> Self {
        Self(0)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Layer {
    pub id: LayerId,
    pub data: Vec<Pixel>,
    pub blend: BlendMode,
    pub file_offset: FilePos,
}

impl Layer {
    pub fn changes(&mut self, other: &Layer) -> Option<LayerDelta> {
        let mut pixel_changes = Vec::new();
        let mut pixels = self.data.iter().enumerate().zip(&other.data);

        for ((index, this), other) in &mut pixels {
            if this != other {
                pixel_changes.reserve_exact(self.data.len());
                pixel_changes.resize(index, 0);
                pixel_changes.push(*this - *other);
                break;
            }
        }

        if pixels.len() == 0 {
            return None;
        }

        pixel_changes.extend(pixels.map(|((_, this), other)| *this - *other));
        self.file_offset = FilePos::default();

        Some(LayerDelta(pixel_changes))
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum BlendMode {
    Average = 0,
}

impl TryFrom<u8> for BlendMode {
    type Error = InvalidBlendMode;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Average),
            _ => Err(InvalidBlendMode),
        }
    }
}

pub struct InvalidBlendMode;

impl From<InvalidBlendMode> for io::Error {
    fn from(_value: InvalidBlendMode) -> Self {
        io::Error::new(io::ErrorKind::InvalidData, "invalid blend mode")
    }
}

#[derive(Debug, Clone)]
pub struct Edit {
    pub when: SystemTime,
    pub op: EditOp,
    pub changes: ImageChanges,
    pub file_offset: FilePos,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum EditOp {
    Tool { name: Symbol, alt: bool },
    NewColor,
}

pub struct InvalidEditOp;

impl From<InvalidEditOp> for io::Error {
    fn from(_value: InvalidEditOp) -> Self {
        io::Error::new(io::ErrorKind::InvalidData, "invalid edit op")
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PaletteChange {
    InsertColor(usize, Color),
    RemoveColor(usize, Color),
    Change(usize, Color),
}

impl PaletteChange {
    pub fn apply(&self, palette: &mut Vec<Color>) {
        match self {
            PaletteChange::InsertColor(index, color) => {
                palette.insert(*index, *color);
            }
            PaletteChange::RemoveColor(index, _) => {
                palette.remove(*index);
            }
            PaletteChange::Change(index, color) => {
                palette[*index] = *color;
            }
        }
    }

    pub fn revert(&self, palette: &mut Vec<Color>) {
        match self {
            PaletteChange::InsertColor(index, _) => {
                palette.remove(*index);
            }
            PaletteChange::RemoveColor(index, color) => {
                palette.insert(*index, *color);
            }
            PaletteChange::Change(index, color) => {
                palette[*index] = *color;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum LayerChange {
    Insert(usize, Layer),
    Change(usize, LayerDelta),
    Remove(usize, Layer),
}

#[derive(Debug, Clone)]
pub struct LayerDelta(Vec<u16>);

impl LayerDelta {
    pub fn apply(&self, layer: &mut Layer) {
        for (delta, pixel) in self.0.iter().copied().zip(&mut layer.data) {
            *pixel += delta;
        }
    }

    pub fn revert(&self, layer: &mut Layer) {
        for (delta, pixel) in self.0.iter().copied().zip(&mut layer.data) {
            *pixel -= delta;
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PaletteColor(pub NonZeroU16);

impl Image {
    pub fn composite(&self, rgba: &mut Vec<u8>, checker_colors: Option<[Color; 2]>) {
        let size = self.size.map(|v| v.get() as usize);
        rgba.clear();
        if let Some(checker_colors) = checker_colors {
            // Create our two patterns
            for x in 0..size.width {
                let color = checker_colors[x / 16 % 2];
                rgba.extend(color.0.to_be_bytes());
            }
            for _ in 1..16.min(size.height) {
                rgba.extend_from_within(0..size.width * 4);
            }
            for _ in 16..32.min(size.height) {
                rgba.extend_from_within(4 * 16..size.width * 4 + 4 * 16);
            }

            // Now we can repeat these patterns until the buffer is full
            for _ in 1..size.height / 32 {
                rgba.extend_from_within(0..size.width * 32 * 4);
            }

            let remaining = size.height % 32;
            if remaining > 0 {
                rgba.extend_from_within(0..remaining * size.width * 4);
            }
        }

        for layer in &self.layers {
            rgba.par_iter_mut()
                .chunks(size.width * 4)
                .zip(layer.data.par_iter().chunks(size.width))
                .for_each(|(mut row, layer_row)| {
                    for (dest, src) in row.chunks_mut(4).zip(layer_row) {
                        if let Some(index) = src.0 {
                            let color = self.palette[usize::from(index.0.get() - 1)];
                            let BlendMode::Average = layer.blend;

                            let alpha = color.alpha();
                            if alpha == 255 {
                                *dest[0] = color.red();
                                *dest[1] = color.green();
                                *dest[2] = color.blue();
                                *dest[3] = 255;
                            } else {
                                *dest[0] =
                                    ((u16::from(color.red()) + u16::from(*dest[0])) / 2) as u8;
                                *dest[1] =
                                    ((u16::from(color.green()) + u16::from(*dest[1])) / 2) as u8;
                                *dest[2] =
                                    ((u16::from(color.blue()) + u16::from(*dest[2])) / 2) as u8;
                                *dest[3] = ((u16::from(alpha) + u16::from(*dest[3])) / 2) as u8;
                            }
                        }
                    }
                })
        }
    }
}

pub struct ImageLayer<'a> {
    image: &'a mut Image,
    layer: usize,
}

impl ImageLayer<'_> {
    pub fn pixel(&self, coord: impl ImageCoordinate) -> Option<Pixel> {
        self.image
            .coordinate_to_offset(coord)
            .map(|index| self.image.layers[self.layer].data[index])
    }

    pub fn pixel_mut(&mut self, coord: impl ImageCoordinate) -> Option<&mut Pixel> {
        self.image
            .coordinate_to_offset(coord)
            .map(|index| &mut self.image.layers[self.layer].data[index])
    }

    pub fn image(&self) -> &Image {
        self.image
    }

    pub fn fill(&mut self, origin: impl ImageCoordinate, color: Pixel) -> bool {
        fn scan(
            this: &mut ImageLayer<'_>,
            lx: u32,
            rx: u32,
            y: u32,
            test_color: Pixel,
            stack: &mut Vec<Point<u32>>,
        ) {
            let mut span_added = false;
            for x in lx..=rx {
                if !this.test_pixel(Point::new(x, y), test_color) {
                    span_added = false;
                } else if !span_added {
                    stack.push(Point::new(x, y));
                    span_added = true;
                }
            }
        }

        let Some(origin) = origin.to_pixel() else {
            return false;
        };
        let Some(test_color) = self.pixel(origin) else {
            return false;
        };
        if test_color == color {
            return false;
        }

        let mut stack = vec![origin];
        while let Some(Point { mut x, y }) = stack.pop() {
            let mut lx = x;
            while lx > 0 && self.compare_swap_pixel(Point::new(lx - 1, y), test_color, color) {
                lx -= 1;
            }
            while self.compare_swap_pixel(Point::new(x, y), test_color, color) {
                x += 1;
            }
            if x > 0 {
                scan(self, lx, x - 1, y + 1, test_color, &mut stack);
                if y > 0 {
                    scan(self, lx, x - 1, y - 1, test_color, &mut stack);
                }
            }
        }

        true
    }

    pub fn test_pixel(&self, coord: impl ImageCoordinate, color: Pixel) -> bool {
        self.pixel(coord) == Some(color)
    }

    pub fn compare_swap_pixel(
        &mut self,
        coord: impl ImageCoordinate,
        swap_if_color: Pixel,
        new_color: Pixel,
    ) -> bool {
        let Some(pixel) = self.pixel_mut(coord) else {
            return false;
        };
        if *pixel == swap_if_color {
            *pixel = new_color;
            true
        } else {
            false
        }
    }
}

pub trait ImageCoordinate {
    // fn to_offset(self, image: &Image) -> Option<usize>;
    fn to_pixel(self) -> Option<Point<u32>>;
}

impl ImageCoordinate for Point<f32> {
    fn to_pixel(self) -> Option<Point<u32>> {
        if self.x.is_sign_positive() && self.y.is_sign_positive() {
            Some(self.map(|c| c.floor() as u32))
        } else {
            None
        }
    }
}

impl ImageCoordinate for Point<u32> {
    fn to_pixel(self) -> Option<Point<u32>> {
        Some(self)
    }
}

impl ImageCoordinate for Point<UPx> {
    fn to_pixel(self) -> Option<Point<u32>> {
        Some(self.map(UPx::get))
    }
}

#[derive(Debug)]
pub struct ColorHistory(VecDeque<Pixel>);

impl ColorHistory {
    pub fn new(initial_color: Pixel) -> Self {
        Self(VecDeque::from_iter([initial_color]))
    }

    pub fn current(&self) -> Pixel {
        self.0[0]
    }

    pub fn previous(&self) -> Option<Pixel> {
        self.0.get(1).copied()
    }

    pub fn push(&mut self, color: Pixel) {
        self.0.retain(|c| *c != color);
        self.0.push_front(color);
    }

    pub fn swap_previous(&mut self) {
        if self.0.len() > 1 {
            self.0.swap(0, 1);
        }
    }

    pub fn cycle_by(&mut self, amount: i16) {
        if amount < 0 {
            self.0
                .rotate_left(usize::try_from(-amount).expect("infallible"));
        } else {
            self.0
                .rotate_right(usize::try_from(amount).expect("infallible"));
        }
    }
}
