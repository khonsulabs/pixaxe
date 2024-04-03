use std::num::NonZeroUsize;

use cushy::figures::Point;
use cushy::value::{Dynamic as CushyDynamic, Source};
use muse::compiler::Compiler;
use muse::refuse::{CollectionGuard, SimpleType};
use muse::symbol::SymbolRef;
use muse::syntax::{SourceCode, SourceId};
use muse::value::{
    CustomType, Dynamic as MuseDynamic, Rooted, RustType, StaticRustFunctionTable,
    Value as MuseValue,
};
use muse::vm::{ExecutionError, Fault, InvokeArgs, Register, Vm};
use pixaxe_core::Pixel;

use crate::EditState;

#[derive(Debug)]
pub struct Editor(CushyDynamic<EditState>);

impl CustomType for Editor {
    fn muse_type(&self) -> &muse::value::TypeRef {
        static TYPE: RustType<Editor> = RustType::new("Editor", |t| {
            t.with_invoke(|_| {
                |this, vm, name, arity| {
                    static FUNCTIONS: StaticRustFunctionTable<Editor> =
                        StaticRustFunctionTable::new(|f| {
                            f.with_fn("current_color", 0, |_vm, this| {
                                Ok(MuseValue::from(
                                    this.0
                                        .map_ref(|state| state.color_history.current().into_u16()),
                                ))
                            })
                            .with_fn("set_pixel", 3, |vm, this| {
                                let x = vm[Register(0)]
                                    .take()
                                    .as_u32()
                                    .ok_or(Fault::ExpectedInteger)?;
                                let y = vm[Register(1)]
                                    .take()
                                    .as_u32()
                                    .ok_or(Fault::ExpectedInteger)?;
                                let new_color = vm[Register(2)]
                                    .take()
                                    .as_u16()
                                    .ok_or(Fault::ExpectedInteger)?;
                                let mut state = this.0.lock();
                                if usize::from(new_color) + 1 >= state.image.palette.len() {
                                    return Err(Fault::OutOfBounds);
                                }
                                let new_color = Pixel::from_u16(new_color);
                                let old_color = state
                                    .image
                                    .layer_mut(0)
                                    .pixel_mut(Point::new(x, y))
                                    .map(|px| std::mem::replace(px, new_color))
                                    .ok_or(Fault::OutOfBounds)?;
                                let changed = new_color != old_color;
                                state.dirty |= changed;

                                Ok(old_color.into_u16().into())
                            })
                            .with_fn("fill", 3, |vm, this| {
                                let x = vm[Register(0)]
                                    .take()
                                    .as_u32()
                                    .ok_or(Fault::ExpectedInteger)?;
                                let y = vm[Register(1)]
                                    .take()
                                    .as_u32()
                                    .ok_or(Fault::ExpectedInteger)?;
                                let new_color = vm[Register(2)]
                                    .take()
                                    .as_u16()
                                    .ok_or(Fault::ExpectedInteger)?;
                                let mut state = this.0.lock();
                                if usize::from(new_color) + 1 >= state.image.palette.len() {
                                    return Err(Fault::OutOfBounds);
                                }
                                let new_color = Pixel::from_u16(new_color);
                                let changed =
                                    state.image.layer_mut(0).fill(Point::new(x, y), new_color);
                                state.dirty |= changed;
                                Ok(changed.into())
                            })
                        });
                    FUNCTIONS.invoke(vm, name, arity, &this)
                }
            })
        });
        &TYPE
    }
}

impl SimpleType for Editor {}

pub struct EditorScript {
    editor: Rooted<Editor>,
    compiler: Compiler,
    vm: Vm,
}

impl EditorScript {
    pub fn new(editor: CushyDynamic<EditState>) -> Self {
        let mut guard = CollectionGuard::acquire();
        let editor = Rooted::new(Editor(editor), &guard);
        let mut compiler = Compiler::default();
        compiler.push(&SourceCode::new(
            pixaxe_core::BUILTIN_SCRIPT,
            SourceId::new(NonZeroUsize::new(1).expect("not zero")),
        ));
        let code = compiler.build(&guard).unwrap();
        let vm = Vm::new(&guard);
        vm.declare(
            "editor",
            MuseValue::Dynamic(editor.as_any_dynamic()),
            &mut guard,
        )
        .expect("error declaring editor");
        vm.execute(&code, &mut guard).unwrap();

        Self {
            editor,
            compiler,
            vm,
        }
    }

    pub fn invoke(
        &self,
        name: impl Into<SymbolRef>,
        params: impl InvokeArgs,
    ) -> Result<MuseValue, ExecutionError> {
        let mut guard = CollectionGuard::acquire();
        let name = name.into();
        self.vm.invoke(&name, params, &mut guard)
    }

    pub fn editor(&self) -> MuseValue {
        MuseValue::Dynamic(self.editor.as_any_dynamic())
    }
}

impl std::fmt::Debug for EditorScript {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EditorScript").finish_non_exhaustive()
    }
}
