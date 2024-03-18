use std::num::NonZeroU16;

use cushy::figures::units::UPx;
use cushy::figures::Size;
use cushy::styles::Color;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

#[derive(Debug)]
pub struct Image {
    pub size: Size<UPx>,
    pub layers: Vec<Layer>,
    pub palette: Vec<Color>,
}

#[derive(Debug)]
pub struct Layer {
    pub data: Vec<Option<PaletteColor>>,
    pub blend: BlendMode,
}

#[derive(Debug)]
pub enum BlendMode {
    Average,
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
                        if let Some(index) = src {
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
