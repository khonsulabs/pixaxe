use cushy::figures::Point;

use super::{ImageState, Tool};
use crate::{EditOp, ImageLayer, Pixel};

#[derive(Debug)]
pub struct Fill;

impl Tool for Fill {
    fn update(
        &mut self,
        location: Point<f32>,
        mut layer: ImageLayer<'_>,
        state: ImageState<'_>,
        _initial: bool,
        alternate: bool,
    ) -> bool {
        let color = if alternate {
            Pixel::clear()
        } else {
            state.color_history.current()
        };
        layer.fill(location, color)
    }

    fn complete(
        &mut self,
        _layer: ImageLayer<'_>,
        _state: ImageState<'_>,
        _alternate: bool,
    ) -> Option<EditOp> {
        Some(EditOp::Paint)
    }
}
