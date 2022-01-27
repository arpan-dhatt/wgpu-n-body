use std::{
    slice,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    }, marker::PhantomData,
};

/// Adds immutable data to a slice sequentially and concurrently.
pub struct SliceAlloc<'a, T> {
    inner: &'a mut [T],
    alloced: Arc<AtomicUsize>,
}

/// Contains a reservation for a given location in the slice allocator. Passing a reservation to a
/// slice allocator that did not issue this reservation is undefined behavior.
#[derive(Debug)]
pub struct Reserve<'a> {
    ix: usize,
    phantom: PhantomData<&'a ()>
}

impl Into<usize> for Reserve<'_> {
    fn into(self) -> usize {
        self.ix
    }
}

impl Into<usize> for &Reserve<'_> {
    fn into(self) -> usize {
        self.ix
    }
}

unsafe impl<T> Sync for SliceAlloc<'_, T> {}

impl<'a, T> SliceAlloc<'a, T> {
    /// Creates a wrapper around a mutable reference, taking exclusive access so data can be
    /// written.
    pub fn wrap(inner: &'a mut [T]) -> SliceAlloc<'a, T> {
        SliceAlloc {
            inner,
            alloced: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Writes a given value to the end of the currently used space in the slice. Returns a
    /// `Reserve` type which can be used to read and write to that location with the guaruntee that
    /// no other thread may be reading or writing it concurrently. **NOTE**: This only holds if
    /// `Reserve` values are only passed to the `SliceAlloc`s that issue it.
    pub fn write(&mut self, value: T) -> Reserve<'a> {
        let ix = self.alloced.fetch_add(1, Ordering::Relaxed);
        Reserve {
            ix,
            phantom: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.alloced.load(Ordering::Relaxed)
    }
}

impl<T> std::ops::Index<Reserve<'_>> for SliceAlloc<'_, T> {
    type Output = T;

    fn index(&self, index: Reserve<'_>) -> &Self::Output {
        if index.ix >= self.alloced.load(Ordering::Relaxed) {
            panic!("Accessing Mutable Memory")
        }
        &self.inner[index.ix]
    }
}

impl<T> std::ops::IndexMut<Reserve<'_>> for SliceAlloc<'_, T> {
    fn index_mut(&mut self, index: Reserve<'_>) -> &mut Self::Output {
        &mut self.inner[<Reserve<'_> as Into<usize>>::into(index)]
    }
}

impl<'a, T> Clone for SliceAlloc<'a, T> {
    fn clone(&self) -> SliceAlloc<'a, T> {
        Self {
            inner: unsafe {
                slice::from_raw_parts_mut(self.inner.as_ptr() as *mut _, self.inner.len())
            },
            alloced: self.alloced.clone(),
        }
    }
}
