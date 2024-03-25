use std::fmt::Debug;

use cushy::figures::Point;

use crate::{ColorHistory, EditOp, ImageLayer, Layer};

mod fill;
mod pencil;
pub use fill::Fill;
pub use pencil::Pencil;

pub struct ImageState<'a> {
    pub color_history: &'a ColorHistory,
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
