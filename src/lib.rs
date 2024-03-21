use std::num::NonZeroU16;
use std::ops::{AddAssign, Sub, SubAssign};
use std::time::SystemTime;

use cushy::figures::units::UPx;
use cushy::figures::Size;
use cushy::styles::Color;
use kempt::Set;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

#[derive(Debug)]
pub struct ImageFile {
    pub image: Image,
    pub history: Vec<Edit>,
    pub undone: Vec<Edit>,
}

#[derive(Debug, Clone)]
pub struct Image {
    pub size: Size<UPx>,
    pub layers: Vec<Layer>,
    pub palette: Vec<Color>,
}

impl Image {
    pub fn changes(&self, other: &Image) -> ImageChanges {
        let mut layer_changes = Vec::new();

        let mut found_layers = Set::new();
        'layers: for (index, layer) in self.layers.iter().enumerate() {
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
                layer_changes.push(LayerChange::InsertLayer(index, layer.clone()));
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
}

#[derive(Debug, Clone)]
pub struct ImageChanges {
    pub palette: Vec<PaletteChange>,
    pub layers: Vec<LayerChange>,
}

impl ImageChanges {
    pub fn apply(&self, image: &mut Image) {
        for layer in &self.layers {
            match layer {
                LayerChange::InsertLayer(index, layer) => {
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
                LayerChange::InsertLayer(index, _) => {
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

#[derive(Debug, Clone)]
pub struct Layer {
    pub id: usize,
    pub data: Vec<Pixel>,
    pub blend: BlendMode,
}

impl Layer {
    pub fn changes(&self, other: &Layer) -> Option<LayerDelta> {
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

        Some(LayerDelta(pixel_changes))
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum BlendMode {
    Average,
}

#[derive(Debug, Clone)]
pub struct Edit {
    pub when: SystemTime,
    pub op: EditOp,
    pub changes: ImageChanges,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum EditOp {
    Paint,
    Erase,
    NewColor,
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
    InsertLayer(usize, Layer),
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
