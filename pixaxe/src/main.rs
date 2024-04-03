use std::cmp::Ordering;
use std::num::NonZeroU32;
use std::sync::OnceLock;
use std::time::SystemTime;

use cushy::context::{EventContext, GraphicsContext, LayoutContext, Trackable, WidgetContext};
use cushy::figures::units::{Lp, Px, UPx};
use cushy::figures::{FloatConversion, IntoSigned, Point, Rect, Round, ScreenScale, Size, Zero};
use cushy::kludgine::app::winit::event::{MouseButton, MouseScrollDelta};
use cushy::kludgine::app::winit::keyboard::Key;
use cushy::kludgine::app::winit::window::CursorIcon;
use cushy::kludgine::shapes::{PathBuilder, Shape, StrokeOptions};
use cushy::kludgine::text::{Text, TextOrigin};
use cushy::kludgine::wgpu::rwh::HasWindowHandle;
use cushy::kludgine::{wgpu, DrawableExt, Texture};
use cushy::styles::{Color, ColorExt};
use cushy::value::{Dynamic, Source};
use cushy::widget::{
    EventHandling, MakeWidget, Widget, WidgetId, WidgetRef, WrapperWidget, HANDLED, IGNORED,
};
use cushy::widgets::color::RgbaPicker;
use cushy::widgets::layers::{OverlayHandle, OverlayLayer};
use cushy::window::{DeviceId, KeyEvent};
use cushy::{ConstraintLimit, ModifiersExt, Run};
use muse::symbol::{Symbol, SymbolRef};
use muse::value::Value as MuseValue;
use pixaxe_core::{ColorHistory, Edit, EditOp, FilePos, Image, ImageFile, Layer, LayerId, Pixel};
use scripting::EditorScript;

mod scripting;

const CHECKER_LIGHT: Color = Color(0xB0B0B0FF);
const CHECKER_DARK: Color = Color(0x404040FF);
const INITIAL_WIDTH: u16 = 128;
const INITIAL_HEIGHT: u16 = 128;
// https://lospec.com/palette-list/sweetie-16
const SWEETIE_16: [Color; 16] = [
    Color(0x1a1c2cff),
    Color(0x5d275dff),
    Color(0xb13e53ff),
    Color(0xef7d57ff),
    Color(0xffcd75ff),
    Color(0xa7f070ff),
    Color(0x38b764ff),
    Color(0x257179ff),
    Color(0x29366fff),
    Color(0x3b5dc9ff),
    Color(0x41a6f6ff),
    Color(0x73eff7ff),
    Color(0xf4f4f4ff),
    Color(0x94b0c2ff),
    Color(0x566c86ff),
    Color(0x333c57ff),
];
const INITIAL_COLOR: Pixel = Pixel::indexed(12);
const INITIAL_BACKGROUND: Pixel = Pixel::indexed(0);

fn main() {
    let file = if let Some(path) = std::env::args().nth(1) {
        ImageFile::load(path.into()).expect("error reading file")
    } else {
        let image = Image {
            size: Size::new(
                UPx::new(u32::from(INITIAL_WIDTH)),
                UPx::new(u32::from(INITIAL_HEIGHT)),
            ),
            layers: vec![Layer {
                id: LayerId::first(),
                data: vec![INITIAL_BACKGROUND; usize::from(INITIAL_WIDTH * INITIAL_HEIGHT)],
                blend: pixaxe_core::BlendMode::Average,
                file_offset: FilePos::default(),
            }],
            palette: SWEETIE_16.to_vec(),
        };
        ImageFile::new(image)
    };

    let overlays = OverlayLayer::default();
    let data = Dynamic::new(EditState {
        image: file.data.image.clone(),
        file,
        texture_and_buffer: OnceLock::new(),
        dirty: true,
        zoom: 2.,
        drag_mode: None,
        rgba: Vec::new(),
        scroll: Point::ZERO,
        max_scroll: Point::ZERO,
        render_size: Size::ZERO,
        hovered: None,
        color_history: ColorHistory::new(INITIAL_COLOR),
        keyboard_mode: KeyboardMode::default(),
        overlays: overlays.clone(),
        overlay: None,
        tools: vec![Symbol::from("pencil"), Symbol::from("fill")],
        selected_tool: 0,
    });
    let area = EditArea {
        state: data.clone(),
        script: EditorScript::new(data.clone()),
    }
    .make_widget();
    let area_id = area.id();

    let palette = Palette::new(data.clone());
    let layers = Root {
        data: data.clone(),
        child: WidgetRef::new(
            area.expand()
                .and(palette.width(Lp::points(129)))
                .into_columns()
                .expand()
                .and(overlays)
                .into_layers(),
        ),
        editor: area_id,
    };
    let mut window = layers.into_window();
    window.vsync = false;
    window.multisample_count = NonZeroU32::MIN;
    window.run().unwrap();
}

#[derive(Debug)]
struct Root {
    data: Dynamic<EditState>,
    child: WidgetRef,
    editor: WidgetId,
}

impl WrapperWidget for Root {
    fn child_mut(&mut self) -> &mut cushy::widget::WidgetRef {
        &mut self.child
    }

    fn adjust_child_constraints(
        &mut self,
        available_space: Size<ConstraintLimit>,
        _context: &mut LayoutContext<'_, '_, '_, '_>,
    ) -> Size<ConstraintLimit> {
        available_space.map(|limit| ConstraintLimit::Fill(limit.max()))
    }

    fn mouse_wheel(
        &mut self,
        _device_id: DeviceId,
        delta: MouseScrollDelta,
        _phase: cushy::kludgine::app::winit::event::TouchPhase,
        context: &mut EventContext<'_>,
    ) -> EventHandling {
        let mut state = self.data.lock();
        if context.modifiers().state().alt_key() {
            // Cycle colors
            let index_delta = match delta {
                MouseScrollDelta::LineDelta(_, y) => y.ceil() as i16,
                MouseScrollDelta::PixelDelta(px) => px.y.ceil() as i16,
            };
            if context.modifiers().state().shift_key() {
                // Cycle through previously selected colors
                state.cycle_color(index_delta);
            } else {
                // Cycle through palette colors
                let mut next_color =
                    i32::from(state.color_history.current().into_u16()) - i32::from(index_delta);
                let palette_len =
                    i32::try_from(state.image.palette.len()).expect("too many colors") + 1;
                while next_color < 0 {
                    next_color += palette_len;
                }
                while next_color >= palette_len {
                    next_color -= palette_len;
                }
                state.color_history.push(Pixel::from_u16(
                    u16::try_from(next_color).expect("too many colors"),
                ));
            }
        } else {
            // Zoom
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
                let y_ratio =
                    new_size.height.into_float() / current_zoomed_size.height.into_float();
                state.scroll.y = state.scroll.y.max(Px::ZERO) * y_ratio + hover_adjust.y;
            }
        }

        context.set_needs_redraw();
        HANDLED
    }

    fn accept_focus(&mut self, _context: &mut EventContext<'_>) -> bool {
        true
    }

    fn focus(&mut self, _context: &mut EventContext<'_>) {
        let mut state = self.data.lock();
        state.keyboard_mode = KeyboardMode::default();
    }

    fn keyboard_input(
        &mut self,
        _device_id: DeviceId,
        input: KeyEvent,
        _is_synthetic: bool,
        context: &mut EventContext<'_>,
    ) -> EventHandling {
        let mut state = self.data.lock();
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
                "v" if context.modifiers().only_primary() => {
                    if input.state.is_pressed() {
                        if let Some(contents) = context
                            .cushy()
                            .clipboard_guard()
                            .and_then(|mut clip| clip.get_text().ok())
                        {
                            let mut updated = false;
                            for line in contents.lines() {
                                let line = line.trim();
                                let line = line.strip_prefix('#').unwrap_or(line);
                                let Ok(parsed) = u32::from_str_radix(line, 16) else {
                                    continue;
                                };
                                let color = if matches!(line.len(), 3 | 6) {
                                    Color(parsed << 8 | 0xFF)
                                } else {
                                    Color(parsed)
                                };
                                state.image.palette.push(color);
                                updated = true;
                            }

                            if updated {
                                state.commit_op(EditOp::NewColor);
                                context.set_needs_redraw();
                            }
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
                                    .previous()
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
                                                let state = self.data.clone();
                                                move |()| {
                                                    let mut state = state.lock();
                                                    let new_index =
                                                        state.image.palette.len() as u16;
                                                    state.image.palette.push(selected_color.get());
                                                    state
                                                        .color_history
                                                        .push(Pixel::indexed(new_index));
                                                    state.commit_op(EditOp::NewColor);
                                                    state.overlays.dismiss_all();
                                                }
                                            }))
                                            .into_rows()
                                            .contain()
                                            .size(Size::new(Lp::inches(4), Lp::inches(4))),
                                    )
                                    .above(self.editor)
                                    .show(),
                            );
                            new_mode = Some(KeyboardMode::Default);
                        } else {
                            state.color_history.push(Pixel::clear());
                            new_mode = Some(KeyboardMode::SelectColor(0));
                        }
                    }
                    HANDLED
                }
                "s" if context.modifiers().primary() => {
                    if input.state.is_pressed() {
                        if context.modifiers().state().shift_key() {
                            save_as(
                                self.data.clone(),
                                context.window().winit().expect("winit handle missing"),
                            );
                        } else if state.file.on_disk() {
                            // TODO don't panic on io error
                            state.file.save().unwrap();
                        } else {
                            save_as(
                                self.data.clone(),
                                context.window().winit().expect("winit handle missing"),
                            );
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
                        state.color_history.swap_previous();
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

fn save_as(state: Dynamic<EditState>, parent: &impl HasWindowHandle) {
    // TODO we should prevent multiple dialogs from being open at the same time
    let dialog = rfd::FileDialog::new()
        .set_file_name("unnamed.paxe")
        .set_parent(parent)
        .add_filter("Pixaxe Image", &["paxe"])
        .set_can_create_directories(true);
    std::thread::spawn(move || {
        if let Some(path) = dialog.save_file() {
            let mut state = state.lock();
            // TODO don't panic on io error
            state.file.save_as(path).unwrap();
        }
    });
}

#[derive(Debug, Clone, Copy)]
struct ToolAction {
    index: usize,
    alt: bool,
}

#[derive(Debug)]
struct EditArea {
    state: Dynamic<EditState>,
    script: EditorScript,
}

impl EditArea {
    fn apply_drag_op(
        &mut self,
        coord: Point<Px>,
        context: &mut WidgetContext<'_>,
        initial: bool,
    ) -> EventHandling {
        let mut state = self.state.lock();
        match state.apply_drag_op(coord, context) {
            Ok(result) => result,
            Err(tool) => {
                let tool_name = &state.tools[tool.index];
                let coord = state.image_coordinate(coord);
                let function_path = format!("{tool_name}.update");
                drop(state);
                if let MuseValue::Bool(true) = self
                    .script
                    .invoke(
                        function_path,
                        [
                            MuseValue::Nil,
                            MuseValue::from(coord.x),
                            MuseValue::from(coord.y),
                            self.script.editor(),
                            MuseValue::from(initial),
                            MuseValue::from(tool.alt),
                        ],
                    )
                    .unwrap()
                {
                    context.set_needs_redraw();
                }
                HANDLED
            }
        }
    }

    fn commit_drag_op(&mut self) {
        let mut state = self.state.lock();
        if let Some(tool) = state.commit_drag_op() {
            let tool_name = state.tools[tool.index].clone();
            let function_path = SymbolRef::from(format!("{tool_name}.complete"));
            drop(state);
            self.script
                .invoke(
                    function_path,
                    [
                        MuseValue::Nil,
                        self.script.editor(),
                        MuseValue::from(tool.alt),
                    ],
                )
                .unwrap();
            self.state.lock().commit_op(EditOp::Tool {
                name: tool_name,
                alt: tool.alt,
            });
        }
    }
}

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
    color_history: ColorHistory,
    keyboard_mode: KeyboardMode,
    overlays: OverlayLayer,
    overlay: Option<OverlayHandle>,
    tools: Vec<Symbol>,
    selected_tool: usize,
}

impl EditState {
    fn image_coordinate(&self, widget_pos: Point<Px>) -> Point<f32> {
        (widget_pos + self.scroll).map(|c| c.into_float() / self.zoom)
    }

    fn apply_drag_op(
        &mut self,
        coord: Point<Px>,
        context: &mut WidgetContext<'_>,
    ) -> Result<EventHandling, ToolAction> {
        if let Some(mode) = self.drag_mode {
            match mode {
                DragMode::ApplyTool(tool) => return Err(tool),
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

            Ok(HANDLED)
        } else {
            Ok(IGNORED)
        }
    }

    fn commit_drag_op(&mut self) -> Option<ToolAction> {
        match self.drag_mode.take() {
            Some(DragMode::ApplyTool(tool)) => Some(tool),
            None | Some(DragMode::Scroll { .. }) => None,
        }
    }

    fn cycle_color(&mut self, amount: i16) {
        self.color_history.cycle_by(amount);
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
        let changes = self.image.changes(&self.file.data.image);
        if !changes.is_empty() {
            self.file.data.undone.clear();

            self.file.data.history.push(Edit {
                when: SystemTime::now(),
                op,
                changes,
                file_offset: FilePos::default(),
            });
            self.file.data.image = self.image.clone();
        }
    }

    fn ensure_consistency(&mut self) {
        if self.color_history.current().index().map_or(false, |index| {
            usize::from(index) >= self.image.palette.len()
        }) {
            self.color_history.push(if self.image.palette.is_empty() {
                Pixel::clear()
            } else {
                Pixel::indexed((self.image.palette.len() - 1) as u16)
            });
        }
    }

    fn undo(&mut self, context: &mut WidgetContext<'_>) {
        if let Some(edit) = self.file.data.history.pop() {
            edit.changes.revert(&mut self.image);
            self.file.data.image = self.image.clone();
            self.file.data.undone.push(edit);
            self.dirty = true;
            context.set_needs_redraw();
            self.ensure_consistency();
        }
    }

    fn redo(&mut self, context: &mut WidgetContext<'_>) {
        if let Some(edit) = self.file.data.undone.pop() {
            edit.changes.apply(&mut self.image);
            self.file.data.image = self.image.clone();
            self.file.data.history.push(edit);
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
                    self.color_history.push(Pixel::from_u16(new_color));
                    self.keyboard_mode = KeyboardMode::SelectColor(new_color);
                } else {
                    self.keyboard_mode = KeyboardMode::Default;
                }
            }
            KeyboardMode::Default => {
                let tool = usize::from(number);
                if tool > 0 && tool <= self.tools.len() {
                    self.selected_tool = tool - 1;
                }
            }
        }
    }
}

impl Widget for EditArea {
    fn redraw(&mut self, context: &mut GraphicsContext<'_, '_, '_, '_>) {
        let mut state = self.state.lock();
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

    fn mouse_down(
        &mut self,
        location: Point<Px>,
        _device_id: DeviceId,
        button: MouseButton,
        context: &mut EventContext<'_>,
    ) -> EventHandling {
        let mut state = self.state.lock();
        state.keyboard_mode = KeyboardMode::default();

        state.drag_mode = match button {
            MouseButton::Right => Some(DragMode::ApplyTool(ToolAction {
                index: state.selected_tool,
                alt: true,
            })),
            MouseButton::Left => Some(DragMode::ApplyTool(ToolAction {
                index: state.selected_tool,
                alt: false,
            })),
            MouseButton::Middle => Some(DragMode::Scroll {
                start_scroll: state.scroll,
                start_location: location,
            }),
            _ => None,
        };

        drop(state);
        self.apply_drag_op(location, context, true)
    }

    fn mouse_drag(
        &mut self,
        location: Point<Px>,
        _device_id: DeviceId,
        _button: MouseButton,
        context: &mut EventContext<'_>,
    ) {
        self.apply_drag_op(location, context, false);
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
        let mut state = self.state.lock();
        state.hovered = Some(location);
        state
            .image
            .coordinate_to_offset(state.image_coordinate(location))
            .map(|_| CursorIcon::Crosshair)
    }

    fn unhover(&mut self, _context: &mut EventContext<'_>) {
        let mut state = self.state.lock();
        state.hovered = None;
    }
}

#[derive(Debug, Clone, Copy)]
enum DragMode {
    ApplyTool(ToolAction),
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

#[derive(Debug)]
struct Palette {
    data: Dynamic<EditState>,
    swatch_size: Lp,
    swatches_per_row: usize,
    size: Size<Lp>,
}

impl Palette {
    fn new(data: Dynamic<EditState>) -> Self {
        Self {
            data,
            swatch_size: Lp::ZERO,
            size: Size::ZERO,
            swatches_per_row: 0,
        }
    }
}

impl Widget for Palette {
    fn redraw(&mut self, context: &mut GraphicsContext<'_, '_, '_, '_>) {
        self.data.redraw_when_changed(context);

        let data = self.data.lock();
        self.size = context.gfx.size().into_lp(context.gfx.scale());
        let selected_index = data.color_history.current().index().map(usize::from);
        self.swatch_size = Lp::points(32); // TODO this should be configurable
        let lp_wide = context.gfx.size().width.into_lp(context.gfx.scale());
        self.swatches_per_row = usize::try_from((lp_wide / self.swatch_size).ceil().get())
            .expect("too big")
            .max(1);
        self.swatch_size = lp_wide / i32::try_from(self.swatches_per_row).expect("too big");

        // Normally we'd used integer math by doing (x + w - 1) / w, but we are
        // injecting 2 more swatches: no color and new color.
        let selected_color = context.theme().surface.outline;
        let rows = (data.image.palette.len() + self.swatches_per_row + 1) / self.swatches_per_row;
        for row in 0..rows {
            let y = self.swatch_size * i32::try_from(row).expect("too big");
            let mut x = Lp::ZERO;
            for col in 0..self.swatches_per_row {
                let index = row * self.swatches_per_row + col;
                let index = match index.cmp(&data.image.palette.len()) {
                    Ordering::Less => Some(index),
                    Ordering::Equal => None,
                    Ordering::Greater => Some(index - 1),
                };

                let selected = index == selected_index;
                let swatch_rect = Rect::new(Point::new(x, y), Size::squared(self.swatch_size));
                let midpoint = swatch_rect.origin + swatch_rect.size / 2;
                x += self.swatch_size;
                if let Some(index) = index {
                    if let Some(color) = data.image.palette.get(index) {
                        if color.alpha() < 255 {
                            draw_checkerboard(swatch_rect, self.swatch_size, context);
                        }
                        context
                            .gfx
                            .draw_shape(&Shape::filled_rect(swatch_rect, *color));
                        let text_color = color.most_contrasting(&[Color::WHITE, Color::BLACK]);
                        context.gfx.draw_text(
                            Text::new(&format!("c{}", index + 1), text_color)
                                .origin(TextOrigin::Center)
                                .translate_by(midpoint),
                        );
                    } else if index == data.image.palette.len() {
                        let add_size = self.swatch_size / 2;
                        let add_width = add_size / 8;
                        let add_color = context.theme().surface.on_color;
                        context.gfx.draw_shape(&Shape::filled_rect(
                            Rect::new(
                                Point::new(midpoint.x - add_size / 2, midpoint.y - add_width / 2),
                                Size::new(add_size, add_width),
                            ),
                            add_color,
                        ));
                        context.gfx.draw_shape(&Shape::filled_rect(
                            Rect::new(
                                Point::new(midpoint.x - add_width / 2, midpoint.y - add_size / 2),
                                Size::new(add_width, add_size),
                            ),
                            add_color,
                        ));
                    }
                } else {
                    draw_checkerboard(swatch_rect, self.swatch_size, context);
                    let (top_left, bottom_right) = swatch_rect.extents();
                    let no_color = Color::RED;
                    let width = Lp::mm(2);
                    context.gfx.draw_shape(
                        &PathBuilder::new(Point::new(bottom_right.x, top_left.y))
                            .line_to(Point::new(bottom_right.x, top_left.y + width))
                            .line_to(Point::new(top_left.x + width, bottom_right.y))
                            .line_to(Point::new(top_left.x, bottom_right.y))
                            .line_to(Point::new(top_left.x, bottom_right.y - width))
                            .line_to(Point::new(bottom_right.x - width, top_left.y))
                            .close()
                            .fill(no_color),
                    );
                }

                if selected {
                    let outline_width = Lp::mm(1);
                    context.gfx.draw_shape(&Shape::stroked_rect(
                        swatch_rect.inset(outline_width / 2),
                        StrokeOptions::lp_wide(outline_width).colored(selected_color),
                    ));
                }
            }
        }
    }

    fn mouse_down(
        &mut self,
        location: Point<Px>,
        _device_id: DeviceId,
        _button: MouseButton,
        context: &mut EventContext<'_>,
    ) -> EventHandling {
        let location = location.into_lp(context.kludgine.scale()) / self.swatch_size;
        let Ok(row) = usize::try_from(location.y.get()) else {
            return IGNORED;
        };
        let Ok(column) = usize::try_from(location.x.get()) else {
            return IGNORED;
        };
        let index = self.swatches_per_row * row + column;
        let mut data = self.data.lock();
        match index.cmp(&data.image.palette.len()) {
            Ordering::Less => {
                data.color_history.push(Pixel::indexed(
                    u16::try_from(index).expect("too many colors"),
                ));
            }
            Ordering::Equal => {
                data.color_history.push(Pixel::clear());
            }
            Ordering::Greater => {
                eprintln!("TODO: Show create color")
            }
        }

        HANDLED
    }

    fn hit_test(&mut self, _location: Point<Px>, _context: &mut EventContext<'_>) -> bool {
        true
    }
}

fn draw_checkerboard(
    swatch_rect: Rect<Lp>,
    swatch_size: Lp,
    context: &mut GraphicsContext<'_, '_, '_, '_>,
) {
    let (top_left, bottom_right) = swatch_rect.extents();
    let center = (bottom_right - top_left) / 2 + top_left;
    context
        .gfx
        .draw_shape(&Shape::filled_rect(swatch_rect, CHECKER_DARK));
    context.gfx.draw_shape(&Shape::filled_rect(
        Rect::new(
            Point::new(center.x, top_left.y),
            Size::squared(swatch_size / 2),
        ),
        CHECKER_LIGHT,
    ));
    context.gfx.draw_shape(&Shape::filled_rect(
        Rect::new(
            Point::new(top_left.x, center.y),
            Size::squared(swatch_size / 2),
        ),
        CHECKER_LIGHT,
    ));
}
