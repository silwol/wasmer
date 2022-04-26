use std::{
    marker::PhantomData,
    sync::atomic::{AtomicU64, Ordering},
};

use wasmer_types::VMExternRef;
use wasmer_vm::{InstanceHandle, VMFunction, VMGlobal, VMMemory, VMTable};

use crate::Store;

/// Unique ID to identify a context.
///
/// Every handle to an object managed by a context also contains the ID of the
/// context. This is used to check that a handle is always used with the
/// correct context.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct ContextId(u64);

impl Default for ContextId {
    // Allocates a unique ID for a new context.
    fn default() -> Self {
        // No overflow checking is needed here: overflowing this would take
        // thousands of years.
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Trait to represent an object managed by a context. This is implemented on
/// the VM types managed by the context.
pub(crate) trait ContextObject: Sized {
    fn list(ctx: &ContextObjects) -> &Vec<Self>;
    fn list_mut(ctx: &mut ContextObjects) -> &mut Vec<Self>;
}
macro_rules! impl_context_object {
    ($($field:ident => $ty:ty,)*) => {
        $(
            impl ContextObject for $ty {
                fn list(ctx: &ContextObjects) -> &Vec<Self> {
                    &ctx.$field
                }
                fn list_mut(ctx: &mut ContextObjects) -> &mut Vec<Self> {
                    &mut ctx.$field
                }
            }
        )*
    };
}
// TODO: Some of these types are definitely wrong and need to be reviewed.
impl_context_object! {
    host_funcs => VMFunction,
    tables => VMTable,
    globals => VMGlobal,
    instances => InstanceHandle,
    memories => VMMemory,
    extern_vals => VMExternRef,
}

/// Handle to an object managed by a context.
///
/// Internally this is just an integer index into a context. A reference to the
/// context must be passed in separately to access the actual object.
pub(crate) struct ContextHandle<T: ContextObject> {
    id: ContextId,
    idx: usize,
    marker: PhantomData<fn() -> T>,
}

impl<T: ContextObject> ContextHandle<T> {
    /// Moves the given object into a context and returns a handle to it.
    pub(crate) fn new(ctx: &mut ContextObjects, val: T) -> Self {
        let list = T::list_mut(ctx);
        let idx = list.len();
        list.push(val);
        ContextHandle {
            id: ctx.id,
            idx,
            marker: PhantomData,
        }
    }

    /// Returns a reference to the object that this handle points to.
    pub(crate) fn get<'a>(&self, ctx: &'a ContextObjects) -> &'a T {
        assert_eq!(self.id, ctx.id, "object used with the wrong context");
        &T::list(ctx)[self.idx]
    }

    /// Returns a mutable reference to the object that this handle points to.
    pub(crate) fn get_mut<'a>(&'a self, ctx: &'a mut ContextObjects) -> &'a mut T {
        assert_eq!(self.id, ctx.id, "object used with the wrong context");
        &mut T::list_mut(ctx)[self.idx]
    }
}

/// Set of objects managed by a context.
// TODO: Implement MemoryUsage?
#[derive(Default)]
pub(crate) struct ContextObjects {
    id: ContextId,
    memories: Vec<VMMemory>,
    tables: Vec<VMTable>,
    globals: Vec<VMGlobal>,
    host_funcs: Vec<VMFunction>,
    instances: Vec<InstanceHandle>,
    extern_vals: Vec<VMExternRef>,
}

/// We require the context to have a fixed memory address for its lifetime since
/// various bits of the VM have raw pointers that point back to it. Hence we
/// wrap the actual context in a box.
pub(crate) struct ContextInner<T> {
    pub(crate) objects: ContextObjects,
    pub(crate) store: Store,
    pub(crate) data: T,
}

/// A context containing a set of WebAssembly instances, along with host state.
///
/// All WebAssembly instances must exist within a context. In the majority of
/// cases each intance will have its own context, but it is possible to have
/// multiple instances in a context when these instances need to interact with
/// each other, for example sharing a memory between instances or calling
/// functions in another instance.
///
/// The lifetimes of run-time WebAssembly objects, notably [`Instance`],
/// [`Memory`], [`Global`], [`Table`] and [`Function`] is tied to a context:
/// the backing memory for these objects is only freed when the context is
/// freed.
///
/// The `T` generic parameter allows arbitrary data to be attached to a context.
/// This data can be accessed using the [`Context::data`] and
/// [`Context::data_mut`] methods. Host functions defined using
/// [`Function::new_with_env`] and [`Function::new_native_with_env`] receive
/// a reference to the context when they are called.
pub struct Context<T> {
    pub(crate) inner: Box<ContextInner<T>>,
}

impl<T> Context<T> {
    /// Creates a new context with the given host state.
    // TODO: Eliminate the Store type and move its functionality into Engine.
    pub fn new(store: &Store, data: T) -> Self {
        Self {
            inner: Box::new(ContextInner {
                objects: Default::default(),
                store: store.clone(),
                data,
            }),
        }
    }

    /// Returns a reference to the host state in this context.
    pub fn data(&self) -> &T {
        &self.inner.data
    }

    /// Returns a mutable- reference to the host state in this context.
    pub fn data_mut(&mut self) -> &mut T {
        &mut self.inner.data
    }

    /// Drops the context and returns the host state that was stored in it.
    pub fn into_data(self) -> T {
        self.inner.data
    }
}

/// A temporary handle to a [`Context`].
pub struct ContextRef<'a, T> {
    inner: &'a ContextInner<T>,
}

impl<T> ContextRef<'_, T> {
    /// Returns a reference to the host state in this context.
    pub fn data(&self) -> &T {
        &self.inner.data
    }
}

/// A temporary handle to a [`Context`].
pub struct ContextRefMut<'a, T> {
    inner: &'a mut ContextInner<T>,
}

impl<T> ContextRefMut<'_, T> {
    /// Returns a reference to the host state in this context.
    pub fn data(&self) -> &T {
        &self.inner.data
    }

    /// Returns a mutable- reference to the host state in this context.
    pub fn data_mut(&mut self) -> &mut T {
        &mut self.inner.data
    }
}

/// Helper trait for a value that is convertible to a [`ContextRef`].
pub trait AsContextRef {
    type Data;

    fn as_context_ref(&self) -> ContextRef<'_, Self::Data>;
}

/// Helper trait for a value that is convertible to a [`ContextRefMut`].
pub trait AsContextRefMut: AsContextRef {
    fn as_context_ref_mut(&mut self) -> ContextRefMut<'_, Self::Data>;
}

impl<T> AsContextRef for Context<T> {
    type Data = T;

    fn as_context_ref(&self) -> ContextRef<'_, Self::Data> {
        ContextRef { inner: &self.inner }
    }
}
impl<T> AsContextRefMut for Context<T> {
    fn as_context_ref_mut(&mut self) -> ContextRefMut<'_, Self::Data> {
        ContextRefMut {
            inner: &mut self.inner,
        }
    }
}
impl<T> AsContextRef for ContextRef<'_, T> {
    type Data = T;

    fn as_context_ref(&self) -> ContextRef<'_, Self::Data> {
        ContextRef { inner: self.inner }
    }
}
impl<T> AsContextRef for ContextRefMut<'_, T> {
    type Data = T;

    fn as_context_ref(&self) -> ContextRef<'_, Self::Data> {
        ContextRef { inner: self.inner }
    }
}
impl<T> AsContextRefMut for ContextRefMut<'_, T> {
    fn as_context_ref_mut(&mut self) -> ContextRefMut<'_, Self::Data> {
        ContextRefMut { inner: self.inner }
    }
}
