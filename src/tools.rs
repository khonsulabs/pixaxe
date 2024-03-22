use std::collections::VecDeque;
use std::fmt::Debug;

use cushy::figures::Point;

use crate::{EditOp, ImageLayer, Layer, Pixel};

mod pencil;
pub use pencil::Pencil;

pub struct ImageState<'a> {
    pub current_color: Pixel,
    pub color_history: &'a VecDeque<Pixel>,
    pub original: &'a Layer,
}

pub trait Tool: Send + Sync + Debug {
    fn update(
        &mut self,
        location: Point<f32>,
        layer: ImageLayer<'_>,
        state: ImageState<'_>,
        initial: bool,
        alternate: bool,
    ) -> bool;
    fn complete(
        &mut self,
        layer: ImageLayer<'_>,
        state: ImageState<'_>,
        alternate: bool,
    ) -> Option<EditOp>;
}
