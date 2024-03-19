use std::sync::OnceLock;
use std::time::SystemTime;

use cushy::context::{EventContext, WidgetContext};
use cushy::figures::units::{Px, UPx};
use cushy::figures::{FloatConversion, IntoSigned, Point, Rect, Size, Zero};
use cushy::kludgine::app::winit::event::{MouseButton, MouseScrollDelta};
use cushy::kludgine::app::winit::keyboard::Key;
use cushy::kludgine::app::winit::window::CursorIcon;
use cushy::kludgine::{wgpu, Texture};
use cushy::styles::Color;
use cushy::widget::{EventHandling, MakeWidget, Widget, HANDLED, IGNORED};
use cushy::window::{DeviceId, KeyEvent};
use cushy::{ModifiersExt, Run};
use pxli::{Edit, EditOp, Image, ImageFile, Layer, Pixel};

const CHECKER_LIGHT: Color = Color(0xB0B0B0FF);
const CHECKER_DARK: Color = Color(0x404040FF);
const INITIAL_WIDTH: u16 = 128;
const INITIAL_HEIGHT: u16 = 128;

fn main() {
    let image = Image {
        size: Size::new(
            UPx::new(u32::from(INITIAL_WIDTH)),
            UPx::new(u32::from(INITIAL_HEIGHT)),
        ),
        layers: vec![Layer {
            id: 0,
            data: vec![Pixel::default(); usize::from(INITIAL_WIDTH * INITIAL_HEIGHT)],
            blend: pxli::BlendMode::Average,
        }],
        palette: vec![Color::RED],
    };

    let mut window = EditArea {
        image: image.clone(),
        file: ImageFile {
            image,
            history: Vec::new(),
            undone: Vec::new(),
        },
        texture_and_buffer: OnceLock::new(),
        dirty: true,
        zoom: 2.,
        drag_mode: None,
        rgba: Vec::new(),
        scroll: Point::ZERO,
        max_scroll: Point::ZERO,
        render_size: Size::ZERO,
        hovered: None,
    }
    .expand()
    .into_window();
    window.vsync = false;
    window.run().unwrap();
}

#[derive(Debug)]
struct EditArea {
    image: Image,
    file: ImageFile,
    texture_and_buffer: OnceLock<(Texture, wgpu::Buffer)>,
    dirty: bool,
    zoom: f32,
    drag_mode: Option<DragMode>,
    rgba: Vec<u8>,
    scroll: Point<Px>,
    max_scroll: Point<Px>,
    render_size: Size<Px>,
    hovered: Option<Point<Px>>,
}

impl EditArea {
    fn image_coordinate(&self, widget_pos: Point<Px>) -> Point<f32> {
        (widget_pos + self.scroll).map(|c| c.into_float() / self.zoom)
    }

    fn image_coordinate_to_offset(&self, coord: Point<f32>) -> Option<usize> {
        if coord.x < 0.
            || coord.x >= self.image.size.width.into_float()
            || coord.y < 0.
            || coord.y >= self.image.size.height.into_float()
        {
            return None;
        }

        let offset = coord.x.floor() + coord.y.floor() * self.image.size.width.into_float();
        if offset >= 0. {
            Some(offset as usize)
        } else {
            None
        }
    }

    fn apply_drag_op(
        &mut self,
        coord: Point<Px>,
        context: &mut WidgetContext<'_>,
    ) -> EventHandling {
        if let Some(mode) = self.drag_mode {
            match mode {
                DragMode::Paint(color) => {
                    let coord = self.image_coordinate(coord);
                    if let Some(offset) = self.image_coordinate_to_offset(coord) {
                        if std::mem::replace(&mut self.image.layers[0].data[offset], color) != color
                        {
                            self.dirty = true;
                            context.set_needs_redraw();
                        }
                    }
                }
                DragMode::Scroll {
                    start_scroll,
                    start_location,
                } => {
                    if self.max_scroll.x > 0 {
                        let delta = coord.x - start_location.x;
                        let new_scroll = (start_scroll.x - delta)
                            .min(self.max_scroll.x)
                            .max(Px::ZERO);
                        if self.scroll.x != new_scroll {
                            self.scroll.x = new_scroll;
                            context.set_needs_redraw();
                        }
                    }

                    if self.max_scroll.y > 0 {
                        let delta = coord.y - start_location.y;
                        let new_scroll = (start_scroll.y - delta)
                            .min(self.max_scroll.y)
                            .max(Px::ZERO);
                        if self.scroll.y != new_scroll {
                            self.scroll.y = new_scroll;
                            context.set_needs_redraw();
                        }
                    }
                }
            }

            HANDLED
        } else {
            IGNORED
        }
    }

    fn commit_drag_op(&mut self) {
        match self.drag_mode.take() {
            Some(DragMode::Paint(color)) => {
                self.commit_op(if color.is_some() {
                    EditOp::Paint
                } else {
                    EditOp::Erase
                });
            }
            None | Some(DragMode::Scroll { .. }) => {}
        }
    }

    fn constrain_scroll(&mut self, scaled_size: Size<Px>) {
        // Constrain scroll
        let scroll_width = scaled_size.width - self.render_size.width;
        self.scroll.x = if scroll_width > 0 {
            self.max_scroll.x = scroll_width;
            self.scroll.x.min(scroll_width).max(Px::ZERO)
        } else {
            self.max_scroll.x = Px::ZERO;
            scroll_width / 2
        };
        let scroll_height = scaled_size.height - self.render_size.height;
        self.scroll.y = if scroll_height > 0 {
            self.max_scroll.y = scroll_height;
            self.scroll.y.min(scroll_height).max(Px::ZERO)
        } else {
            self.max_scroll.y = Px::ZERO;
            scroll_height / 2
        };
    }

    fn commit_op(&mut self, op: EditOp) {
        let changes = self.image.changes(&self.file.image);
        if !changes.is_empty() {
            self.file.undone.clear();

            self.file.history.push(Edit {
                when: SystemTime::now(),
                op,
                changes,
            });
            self.file.image = self.image.clone();
        }
    }

    fn undo(&mut self, context: &mut WidgetContext<'_>) {
        if let Some(edit) = self.file.history.pop() {
            edit.changes.revert(&mut self.image);
            self.file.image = self.image.clone();
            self.file.undone.push(edit);
            self.dirty = true;
            context.set_needs_redraw();
        }
    }

    fn redo(&mut self, context: &mut WidgetContext<'_>) {
        if let Some(edit) = self.file.undone.pop() {
            edit.changes.apply(&mut self.image);
            self.file.image = self.image.clone();
            self.file.history.push(edit);
            self.dirty = true;
            context.set_needs_redraw();
        }
    }
}

impl Widget for EditArea {
    fn redraw(&mut self, context: &mut cushy::context::GraphicsContext<'_, '_, '_, '_>) {
        if self.dirty {
            self.image
                .composite(&mut self.rgba, Some([CHECKER_DARK, CHECKER_LIGHT]));

            let (texture, buffer) =
                self.texture_and_buffer.get_or_init(|| {
                    self.dirty = false;
                    let buffer = context.gfx.inner_graphics().device().create_buffer(
                        &wgpu::BufferDescriptor {
                            label: None,
                            size: self.rgba.len() as u64,
                            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                            mapped_at_creation: false,
                        },
                    );
                    let texture = Texture::new_with_data(
                        context.gfx.inner_graphics(),
                        self.image.size,
                        wgpu::TextureFormat::Rgba8UnormSrgb,
                        wgpu::TextureUsages::TEXTURE_BINDING,
                        wgpu::FilterMode::Nearest,
                        &self.rgba,
                    );

                    (texture, buffer)
                });
            let gfx = context.gfx.inner_graphics();
            gfx.queue().write_buffer(buffer, 0, &self.rgba);
            let mut encoder = gfx
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.copy_buffer_to_texture(
                wgpu::ImageCopyBuffer {
                    buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(self.image.size.width.get() * 4),
                        rows_per_image: None,
                    },
                },
                wgpu::ImageCopyTexture {
                    texture: texture.wgpu(),
                    mip_level: 0,
                    origin: wgpu::Origin3d::default(),
                    aspect: wgpu::TextureAspect::default(),
                },
                self.image.size.into(),
            );
            gfx.queue().submit([encoder.finish()]);
            self.dirty = false;
        }

        // Constrain scroll
        self.render_size = context.gfx.size().into_signed();
        let scaled_size = self.image.size.into_signed() * self.zoom;
        self.constrain_scroll(scaled_size);
        let scroll_width = scaled_size.width - self.render_size.width;
        self.scroll.x = if scroll_width > 0 {
            self.max_scroll.x = scroll_width;
            self.scroll.x.min(scroll_width).max(Px::ZERO)
        } else {
            self.max_scroll.x = Px::ZERO;
            scroll_width / 2
        };
        let scroll_height = scaled_size.height - self.render_size.height;
        self.scroll.y = if scroll_height > 0 {
            self.max_scroll.y = scroll_height;
            self.scroll.y.min(scroll_height).max(Px::ZERO)
        } else {
            self.max_scroll.y = Px::ZERO;
            scroll_height / 2
        };

        let render_area = Rect::new(-self.scroll, scaled_size);

        context.gfx.draw_texture(
            &self.texture_and_buffer.get().expect("initialized above").0,
            render_area,
        );
    }

    fn mouse_wheel(
        &mut self,
        _device_id: DeviceId,
        delta: MouseScrollDelta,
        _phase: cushy::kludgine::app::winit::event::TouchPhase,
        context: &mut EventContext<'_>,
    ) -> EventHandling {
        let delta_y = match delta {
            MouseScrollDelta::LineDelta(_, y) => y * 10.,
            MouseScrollDelta::PixelDelta(px) => px.y as f32,
        };

        let current_zoomed_size = self.image.size * self.zoom;
        let focal_point = self
            .hovered
            .unwrap_or_else(|| Point::from(self.render_size / 2));
        let current_hover_unscaled = focal_point / self.zoom;
        let zoom_amount = self.zoom * delta_y / 100.;
        self.zoom += zoom_amount;
        self.zoom = (self.zoom * 10.).round() / 10.;
        let new_size = self.image.size * self.zoom;

        let hover_adjust = current_hover_unscaled * self.zoom - focal_point;

        if new_size.width.into_signed() > self.render_size.width {
            let x_ratio = new_size.width.into_float() / current_zoomed_size.width.into_float();
            self.scroll.x = self.scroll.x.max(Px::ZERO) * x_ratio + hover_adjust.x;
        }

        if new_size.height.into_signed() > self.render_size.height {
            let y_ratio = new_size.height.into_float() / current_zoomed_size.height.into_float();
            self.scroll.y = self.scroll.y.max(Px::ZERO) * y_ratio + hover_adjust.y;
        }

        context.set_needs_redraw();
        HANDLED
    }

    fn mouse_down(
        &mut self,
        location: Point<Px>,
        _device_id: DeviceId,
        button: MouseButton,
        context: &mut EventContext<'_>,
    ) -> EventHandling {
        context.focus();

        self.drag_mode = match button {
            MouseButton::Right => Some(DragMode::Paint(Pixel::clear())),
            MouseButton::Left => Some(DragMode::Paint(Pixel::indexed(0))),
            MouseButton::Middle => Some(DragMode::Scroll {
                start_scroll: self.scroll,
                start_location: location,
            }),
            _ => None,
        };

        self.apply_drag_op(location, context)
    }

    fn mouse_drag(
        &mut self,
        location: Point<Px>,
        _device_id: DeviceId,
        _button: MouseButton,
        context: &mut EventContext<'_>,
    ) {
        self.apply_drag_op(location, context);
    }

    fn mouse_up(
        &mut self,
        _location: Option<Point<Px>>,
        _device_id: DeviceId,
        _button: MouseButton,
        _context: &mut EventContext<'_>,
    ) {
        self.commit_drag_op();
    }

    fn hit_test(&mut self, _location: Point<Px>, _context: &mut EventContext<'_>) -> bool {
        true
    }

    fn hover(
        &mut self,
        location: Point<Px>,
        _context: &mut EventContext<'_>,
    ) -> Option<CursorIcon> {
        self.hovered = Some(location);
        self.image_coordinate_to_offset(self.image_coordinate(location))
            .map(|_| CursorIcon::Crosshair)
    }

    fn unhover(&mut self, _context: &mut EventContext<'_>) {
        self.hovered = None;
    }

    fn accept_focus(&mut self, _context: &mut EventContext<'_>) -> bool {
        true
    }

    fn keyboard_input(
        &mut self,
        _device_id: DeviceId,
        input: KeyEvent,
        _is_synthetic: bool,
        context: &mut EventContext<'_>,
    ) -> EventHandling {
        match &input.logical_key {
            Key::Character(c)
                if c.as_ref() == "z"
                    || c.as_ref() == "Z" && context.modifiers().state().primary() =>
            {
                if input.state.is_pressed() {
                    if c.as_ref() == "Z" {
                        self.redo(context);
                    } else {
                        self.undo(context);
                    }
                }
                HANDLED
            }
            _ => IGNORED,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum DragMode {
    Paint(Pixel),
    Scroll {
        start_scroll: Point<Px>,
        start_location: Point<Px>,
    },
}
