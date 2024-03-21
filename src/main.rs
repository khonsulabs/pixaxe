use std::collections::VecDeque;
use std::sync::OnceLock;
use std::time::SystemTime;

use cushy::context::{EventContext, WidgetContext};
use cushy::figures::units::{Lp, Px, UPx};
use cushy::figures::{FloatConversion, IntoSigned, Point, Rect, Size, Zero};
use cushy::kludgine::app::winit::event::{MouseButton, MouseScrollDelta};
use cushy::kludgine::app::winit::keyboard::Key;
use cushy::kludgine::app::winit::window::CursorIcon;
use cushy::kludgine::{wgpu, Texture};
use cushy::styles::Color;
use cushy::value::{Dynamic, Source};
use cushy::widget::{EventHandling, MakeWidget, Widget, HANDLED, IGNORED};
use cushy::widgets::color::RgbaPicker;
use cushy::widgets::layers::{OverlayHandle, OverlayLayer};
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

    let overlays = OverlayLayer::default();
    let area = EditArea(Dynamic::new(EditState {
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
        current_color: Pixel::indexed(0),
        color_history: VecDeque::new(),
        keyboard_mode: KeyboardMode::default(),
        overlays: overlays.clone(),
        overlay: None,
    }));

    let layers = area.expand().and(overlays).into_layers();
    let mut window = layers.into_window();
    window.vsync = false;
    window.run().unwrap();
}

#[derive(Debug, Clone)]
struct EditArea(Dynamic<EditState>);

#[derive(Debug)]
struct EditState {
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
    current_color: Pixel,
    color_history: VecDeque<Pixel>,
    keyboard_mode: KeyboardMode,
    overlays: OverlayLayer,
    overlay: Option<OverlayHandle>,
}

impl EditState {
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

    fn ensure_consistency(&mut self) {
        if self.current_color.index().map_or(false, |index| {
            usize::from(index) >= self.image.palette.len()
        }) {
            self.current_color = if self.image.palette.is_empty() {
                Pixel::clear()
            } else {
                Pixel::indexed((self.image.palette.len() - 1) as u16)
            };
        }
    }

    fn undo(&mut self, context: &mut WidgetContext<'_>) {
        if let Some(edit) = self.file.history.pop() {
            edit.changes.revert(&mut self.image);
            self.file.image = self.image.clone();
            self.file.undone.push(edit);
            self.dirty = true;
            context.set_needs_redraw();
            self.ensure_consistency();
        }
    }

    fn redo(&mut self, context: &mut WidgetContext<'_>) {
        if let Some(edit) = self.file.undone.pop() {
            edit.changes.apply(&mut self.image);
            self.file.image = self.image.clone();
            self.file.history.push(edit);
            self.dirty = true;
            context.set_needs_redraw();
            self.ensure_consistency();
        }
    }

    fn handle_numeric_input(&mut self, number: u8) {
        match self.keyboard_mode {
            KeyboardMode::SelectColor(picked_color) => {
                let new_color = picked_color * 10 + u16::from(number);
                if usize::from(new_color) <= self.image.palette.len() {
                    self.current_color = Pixel::from_u16(new_color);
                    self.keyboard_mode = KeyboardMode::SelectColor(new_color);
                } else {
                    self.keyboard_mode = KeyboardMode::Default;
                }
            }
            KeyboardMode::Default => {}
        }
    }
}

impl Widget for EditArea {
    fn redraw(&mut self, context: &mut cushy::context::GraphicsContext<'_, '_, '_, '_>) {
        let mut state = self.0.lock();
        if state.dirty {
            let state = &mut *state;
            state
                .image
                .composite(&mut state.rgba, Some([CHECKER_DARK, CHECKER_LIGHT]));

            let (texture, buffer) =
                state.texture_and_buffer.get_or_init(|| {
                    state.dirty = false;
                    let buffer = context.gfx.inner_graphics().device().create_buffer(
                        &wgpu::BufferDescriptor {
                            label: None,
                            size: state.rgba.len() as u64,
                            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                            mapped_at_creation: false,
                        },
                    );
                    let texture = Texture::new_with_data(
                        context.gfx.inner_graphics(),
                        state.image.size,
                        wgpu::TextureFormat::Rgba8UnormSrgb,
                        wgpu::TextureUsages::TEXTURE_BINDING,
                        wgpu::FilterMode::Nearest,
                        &state.rgba,
                    );

                    (texture, buffer)
                });
            let gfx = context.gfx.inner_graphics();
            gfx.queue().write_buffer(buffer, 0, &state.rgba);
            let mut encoder = gfx
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.copy_buffer_to_texture(
                wgpu::ImageCopyBuffer {
                    buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(state.image.size.width.get() * 4),
                        rows_per_image: None,
                    },
                },
                wgpu::ImageCopyTexture {
                    texture: texture.wgpu(),
                    mip_level: 0,
                    origin: wgpu::Origin3d::default(),
                    aspect: wgpu::TextureAspect::default(),
                },
                state.image.size.into(),
            );
            gfx.queue().submit([encoder.finish()]);
            state.dirty = false;
        }

        // Constrain scroll
        state.render_size = context.gfx.size().into_signed();
        let scaled_size = state.image.size.into_signed() * state.zoom;
        state.constrain_scroll(scaled_size);
        let scroll_width = scaled_size.width - state.render_size.width;
        state.scroll.x = if scroll_width > 0 {
            state.max_scroll.x = scroll_width;
            state.scroll.x.min(scroll_width).max(Px::ZERO)
        } else {
            state.max_scroll.x = Px::ZERO;
            scroll_width / 2
        };
        let scroll_height = scaled_size.height - state.render_size.height;
        state.scroll.y = if scroll_height > 0 {
            state.max_scroll.y = scroll_height;
            state.scroll.y.min(scroll_height).max(Px::ZERO)
        } else {
            state.max_scroll.y = Px::ZERO;
            scroll_height / 2
        };

        let render_area = Rect::new(-state.scroll, scaled_size);

        context.gfx.draw_texture(
            &state.texture_and_buffer.get().expect("initialized above").0,
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
        let mut state = self.0.lock();
        state.keyboard_mode = KeyboardMode::default();
        let delta_y = match delta {
            MouseScrollDelta::LineDelta(_, y) => y * 10.,
            MouseScrollDelta::PixelDelta(px) => px.y as f32,
        };

        let current_zoomed_size = state.image.size * state.zoom;
        let focal_point = state
            .hovered
            .unwrap_or_else(|| Point::from(state.render_size / 2));
        let current_hover_unscaled = focal_point / state.zoom;
        let zoom_amount = state.zoom * delta_y / 100.;
        state.zoom += zoom_amount;
        state.zoom = (state.zoom * 10.).round() / 10.;
        let new_size = state.image.size * state.zoom;

        let hover_adjust = current_hover_unscaled * state.zoom - focal_point;

        if new_size.width.into_signed() > state.render_size.width {
            let x_ratio = new_size.width.into_float() / current_zoomed_size.width.into_float();
            state.scroll.x = state.scroll.x.max(Px::ZERO) * x_ratio + hover_adjust.x;
        }

        if new_size.height.into_signed() > state.render_size.height {
            let y_ratio = new_size.height.into_float() / current_zoomed_size.height.into_float();
            state.scroll.y = state.scroll.y.max(Px::ZERO) * y_ratio + hover_adjust.y;
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
        let mut state = self.0.lock();
        context.focus();
        state.keyboard_mode = KeyboardMode::default();

        state.drag_mode = match button {
            MouseButton::Right => Some(DragMode::Paint(Pixel::clear())),
            MouseButton::Left => Some(DragMode::Paint(state.current_color)),
            MouseButton::Middle => Some(DragMode::Scroll {
                start_scroll: state.scroll,
                start_location: location,
            }),
            _ => None,
        };

        state.apply_drag_op(location, context)
    }

    fn mouse_drag(
        &mut self,
        location: Point<Px>,
        _device_id: DeviceId,
        _button: MouseButton,
        context: &mut EventContext<'_>,
    ) {
        let mut state = self.0.lock();
        state.apply_drag_op(location, context);
    }

    fn mouse_up(
        &mut self,
        _location: Option<Point<Px>>,
        _device_id: DeviceId,
        _button: MouseButton,
        _context: &mut EventContext<'_>,
    ) {
        let mut state = self.0.lock();
        state.commit_drag_op();
    }

    fn hit_test(&mut self, _location: Point<Px>, _context: &mut EventContext<'_>) -> bool {
        true
    }

    fn hover(
        &mut self,
        location: Point<Px>,
        _context: &mut EventContext<'_>,
    ) -> Option<CursorIcon> {
        let mut state = self.0.lock();
        state.hovered = Some(location);
        state
            .image_coordinate_to_offset(state.image_coordinate(location))
            .map(|_| CursorIcon::Crosshair)
    }

    fn unhover(&mut self, _context: &mut EventContext<'_>) {
        let mut state = self.0.lock();
        state.hovered = None;
    }

    fn accept_focus(&mut self, _context: &mut EventContext<'_>) -> bool {
        true
    }

    fn focus(&mut self, _context: &mut EventContext<'_>) {
        let mut state = self.0.lock();
        state.keyboard_mode = KeyboardMode::default();
    }

    fn keyboard_input(
        &mut self,
        _device_id: DeviceId,
        input: KeyEvent,
        _is_synthetic: bool,
        context: &mut EventContext<'_>,
    ) -> EventHandling {
        let mut state = self.0.lock();
        let state = &mut *state;
        let mut new_mode = None;
        let result = match &input.logical_key {
            Key::Character(c) => match c.as_ref() {
                "z" | "Z" if context.modifiers().state().primary() => {
                    if input.state.is_pressed() {
                        if c.as_ref() == "Z" {
                            state.redo(context);
                        } else {
                            state.undo(context);
                        }
                    }
                    HANDLED
                }
                "c" if !context.modifiers().possible_shortcut() => {
                    if input.state.is_pressed() {
                        if let KeyboardMode::SelectColor(0) = state.keyboard_mode {
                            let selected_color = Dynamic::new(
                                state
                                    .color_history
                                    .front()
                                    .and_then(|px| px.index())
                                    .map_or(Color::WHITE, |index| {
                                        state.image.palette[usize::from(index)]
                                    }),
                            );
                            state.overlay = Some(
                                state
                                    .overlays
                                    .build_overlay(
                                        RgbaPicker::new(selected_color.clone())
                                            .and("Create".into_button().on_click({
                                                let state = self.0.clone();
                                                move |()| {
                                                    let mut state = state.lock();
                                                    let new_index =
                                                        state.image.palette.len() as u16;
                                                    state.image.palette.push(selected_color.get());
                                                    state.current_color = Pixel::indexed(new_index);
                                                    state.commit_op(EditOp::NewColor);
                                                    state.overlays.dismiss_all();
                                                }
                                            }))
                                            .into_rows()
                                            .contain()
                                            .size(Size::new(Lp::inches(4), Lp::inches(4))),
                                    )
                                    .above(context.widget().id())
                                    .show(),
                            );
                            new_mode = Some(KeyboardMode::Default);
                        } else {
                            let mut index = 0;
                            let current_color = state.current_color;
                            while let Some(color) = state.color_history.get(index) {
                                if *color == current_color {
                                    state.color_history.remove(index);
                                } else {
                                    index += 1;
                                }
                            }
                            state.color_history.insert(0, current_color);
                            state.current_color = Pixel::clear();
                            new_mode = Some(KeyboardMode::SelectColor(0));
                        }
                    }
                    HANDLED
                }
                "0" if !context.modifiers().possible_shortcut() => {
                    if input.state.is_pressed() {
                        state.handle_numeric_input(0)
                    }
                    HANDLED
                }
                "1" if !context.modifiers().possible_shortcut() => {
                    if input.state.is_pressed() {
                        state.handle_numeric_input(1)
                    }
                    HANDLED
                }
                "2" if !context.modifiers().possible_shortcut() => {
                    if input.state.is_pressed() {
                        state.handle_numeric_input(2)
                    }
                    HANDLED
                }
                "3" if !context.modifiers().possible_shortcut() => {
                    if input.state.is_pressed() {
                        state.handle_numeric_input(3)
                    }
                    HANDLED
                }
                "4" if !context.modifiers().possible_shortcut() => {
                    if input.state.is_pressed() {
                        state.handle_numeric_input(4)
                    }
                    HANDLED
                }
                "5" if !context.modifiers().possible_shortcut() => {
                    if input.state.is_pressed() {
                        state.handle_numeric_input(5)
                    }
                    HANDLED
                }
                "6" if !context.modifiers().possible_shortcut() => {
                    if input.state.is_pressed() {
                        state.handle_numeric_input(6)
                    }
                    HANDLED
                }
                "7" if !context.modifiers().possible_shortcut() => {
                    if input.state.is_pressed() {
                        state.handle_numeric_input(7)
                    }
                    HANDLED
                }
                "8" if !context.modifiers().possible_shortcut() => {
                    if input.state.is_pressed() {
                        state.handle_numeric_input(8)
                    }
                    HANDLED
                }
                "9" if !context.modifiers().possible_shortcut() => {
                    if input.state.is_pressed() {
                        state.handle_numeric_input(9)
                    }
                    HANDLED
                }
                "x" if !context.modifiers().possible_shortcut() => {
                    if input.state.is_pressed() {
                        if let Some(color) = state.color_history.pop_front() {
                            state.color_history.push_front(state.current_color);
                            state.current_color = color;
                        }
                    }
                    HANDLED
                }
                _ => IGNORED,
            },
            _ => IGNORED,
        };

        if let Some(mode) = new_mode {
            state.keyboard_mode = mode;
        }

        result
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

#[derive(Default, Debug, Clone, Copy)]
enum KeyboardMode {
    #[default]
    Default,
    SelectColor(u16),
}
