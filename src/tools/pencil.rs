use cushy::figures::Point;

use super::{ImageState, Tool};
use crate::{EditOp, ImageLayer, Pixel};

#[derive(Debug)]
pub struct Pencil;

impl Tool for Pencil {
    fn update(
        &mut self,
        location: Point<f32>,
        mut layer: ImageLayer<'_>,
        state: ImageState<'_>,
        alternate: bool,
        _initial: bool,
    ) -> bool {
        if let Some(pixel) = layer.pixel_mut(location) {
            let color = if alternate {
                Pixel::clear()
            } else {
                state.color_history.current()
            };

            return std::mem::replace(pixel, color) != color;
        }
        false
    }

    fn complete(
        &mut self,
        _layer: ImageLayer<'_>,
        _state: ImageState<'_>,
        alternate: bool,
    ) -> Option<EditOp> {
        Some(if alternate {
            EditOp::Erase
        } else {
            EditOp::Paint
        })
    }
}
