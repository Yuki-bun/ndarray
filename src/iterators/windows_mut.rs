use std::marker::PhantomData;
use std::ops::Index;
use std::ops::RangeInclusive;
use std::ptr::NonNull;

use super::Baseiter;
use crate::imp_prelude::*;
use crate::IntoDimension;
use crate::Layout;
use crate::NdIndex;
use crate::NdProducer;
use crate::Slice;

/// Window producer and iterable
///
/// See [`.windows()`](crate::ArrayRef::windows) for more
/// information.
pub struct WindowsMut<'a, A, D> {
    base: RawArrayViewMut<A, D>,
    life: PhantomData<&'a A>,
    window: D,
    strides: D,
    // TODO: figure out proper type
    mut_range: Vec<RangeInclusive<Ix>>,
}

impl<'a, A, D: Dimension> WindowsMut<'a, A, D> {
    pub(crate) fn new<E>(a: ArrayViewMut<'a, A, D>, window_size: E, axis_strides: E) -> Self
    where
        E: IntoDimension<Dim = D>,
    {
        let window = window_size.into_dimension();

        let strides = axis_strides.into_dimension();
        let mut_range = build_mut_range(&window, &strides);
        let window_strides = a.parts.strides.clone();

        let base = build_base(a, window.clone(), strides);
        WindowsMut {
            base: base.into_raw_view_mut(),
            life: PhantomData,
            window,
            strides: window_strides,
            mut_range,
        }
    }
}

pub struct MutWindow<'a, A, D> {
    base: RawArrayViewMut<A, D>,
    mut_range: Vec<RangeInclusive<Ix>>,
    life: PhantomData<&'a A>,
}

impl<'a, A, D, I> Index<I> for MutWindow<'a, A, D>
where
    D: Dimension,
    I: NdIndex<D>,
{
    type Output = A;

    fn index(&self, index: I) -> &Self::Output {
        let view = unsafe { ArrayView::new_(self.base.as_ptr(), self.base.dim(), self.base.strides()) };
        Index::index(&view, index)
    }
}

impl<'a, A, D: Dimension> MutWindow<'a, A, D> {
    unsafe fn new_(ptr: *mut A, dim: D, strides: D, mut_range: Vec<RangeInclusive<Ix>>) -> Self {
        Self {
            base: RawArrayViewMut::new(NonNull::new(ptr).unwrap(), dim, strides),
            mut_range,
            life: PhantomData,
        }
    }
}

impl_ndproducer! {
    ['a, A, D: Dimension]
    [Clone => 'a, A, D: Clone ]
    WindowsMut {
        base,
        life,
        window,
        strides,
        mut_range,
    }
    WindowsMut<'a, A, D> {
        type Item = MutWindow<'a, A, D>;
        type Dim = D;

        unsafe fn item(&self, ptr) {
            MutWindow::new_(ptr, self.window.clone(),
                            self.strides.clone(),
                            self.mut_range.clone())
        }
    }
}

impl<'a, A, D> IntoIterator for WindowsMut<'a, A, D>
where
    D: Dimension,
    A: 'a,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = WindowsIter<'a, A, D>;
    fn into_iter(self) -> Self::IntoIter {
        WindowsIter {
            iter: self.base.into_base_iter(),
            life: self.life,
            window: self.window,
            strides: self.strides,
        }
    }
}

/// Window iterator.
///
/// See [`.windows()`](crate::ArrayRef::windows) for more
/// information.
pub struct WindowsIter<'a, A, D> {
    iter: Baseiter<A, D>,
    life: PhantomData<&'a A>,
    window: D,
    strides: D,
}

impl_iterator! {
    ['a, A, D: Dimension]
    [Clone => 'a, A, D: Clone]
    WindowsIter {
        iter,
        life,
        window,
        strides,
    }
    WindowsIter<'a, A, D> {
        type Item = ArrayView<'a, A, D>;

        fn item(&mut self, ptr) {
            unsafe {
                ArrayView::new(
                    ptr,
                    self.window.clone(),
                    self.strides.clone())
            }
        }
    }
}

send_sync_read_only!(WindowsMut);
send_sync_read_only!(WindowsIter);

fn build_mut_range<D: Dimension>(window: &D, strides: &D) -> Vec<RangeInclusive<Ix>> {
    window
        .slice()
        .iter()
        .zip(strides.slice().iter())
        .map(|(win_size, stride)| {
            let start = win_size - stride;
            let end = stride - 1;
            start..=end
        })
        .collect()
}

/// build the base array of the `Windows` and `AxisWindows` structs
fn build_base<A, D>(a: ArrayViewMut<A, D>, window: D, strides: D) -> ArrayViewMut<A, D>
where
    D: Dimension,
{
    ndassert!(
        a.ndim() == window.ndim(),
        concat!("Window dimension {} does not match array dimension {} ", "(with array of shape {:?})"),
        window.ndim(),
        a.ndim(),
        a.shape()
    );

    ndassert!(
        a.ndim() == strides.ndim(),
        concat!("Stride dimension {} does not match array dimension {} ", "(with array of shape {:?})"),
        strides.ndim(),
        a.ndim(),
        a.shape()
    );

    let mut base = a;
    base.slice_each_axis_inplace(|ax_desc| {
        let len = ax_desc.len;
        let wsz = window[ax_desc.axis.index()];
        let stride = strides[ax_desc.axis.index()];

        if len < wsz {
            Slice::new(0, Some(0), 1)
        } else {
            Slice::new(0, Some((len - wsz + 1) as isize), stride as isize)
        }
    });
    base
}
