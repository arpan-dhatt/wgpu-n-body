use std::{
    sync::{atomic::AtomicUsize, Arc},
    thread,
};

fn execute_coordinated<F>(fs: Vec<F>)
where
    F: FnOnce(Arc<AtomicUsize>) + Send + 'static,
{
    let active_counter = Arc::new(AtomicUsize::new(0));
    let thread_count = fs.len();
    fs.into_iter()
        .map(|c| {
            let ac = active_counter.clone();
            thread::spawn(move || c(ac))
        })
        .map(|t| {
            t.join().unwrap();
        })
        .for_each(drop);
}
