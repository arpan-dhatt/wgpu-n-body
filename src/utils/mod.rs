pub mod slice_alloc;
pub mod coordinated_pool;

pub unsafe fn cast_slice<'a, A, B>(a: &'a [A]) -> &'a [B] {
    let new_size = a.len() * std::mem::size_of::<A>() / std::mem::size_of::<B>();
    std::slice::from_raw_parts(a.as_ptr() as *const _, new_size)
}

pub unsafe fn cast_slice_mut<'a, A, B>(a: &'a mut [A]) -> &'a mut [B] {
    let new_size = a.len() * std::mem::size_of::<A>() / std::mem::size_of::<B>();
    std::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut _, new_size)
}
