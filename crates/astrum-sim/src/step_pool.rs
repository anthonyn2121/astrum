//! A fixed set of persistent threads that run one *shard* each per call —
//! shard `i` always on the same OS thread — for per-tick work too short for a
//! general-purpose pool.
//!
//! Why not rayon: one JSBSim step takes ~5 µs, a rayon `install` costs 3–25 µs
//! of sleep/wake per tick, and work stealing hands each FDM to a different core
//! every tick, so its working set is never in that core's cache. Measured: four
//! F-16s stepped no faster on 4 rayon threads than on one (23 µs/tick), and in
//! 10 µs on 4 shard workers. Here each FDM stays on its worker; idle workers
//! spin for [`SPIN`] (an unpaced run issues the next tick within microseconds)
//! and then park, so a paced run costs no idle CPU.

use std::cell::UnsafeCell;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// How long an idle worker spins before parking, and the caller spins waiting
/// for the workers before yielding.
pub const SPIN: Duration = Duration::from_micros(50);

/// A type-erased `(shards, f)` borrowed from [`ShardPool::run`]'s stack frame.
#[derive(Clone, Copy)]
struct Job {
    data: *const (),
    call: unsafe fn(*const (), usize),
}

struct Shared {
    /// Bumped (`Release`) after `job` is written; a worker runs once per value.
    generation: AtomicU64,
    /// Workers that have not finished the current job.
    pending: AtomicUsize,
    job: UnsafeCell<Option<Job>>,
    panicked: AtomicBool,
    shutdown: AtomicBool,
}

// SAFETY: `job` is written only by `ShardPool::run` while every worker is idle
// (`pending == 0`, observed with `Acquire`), and read by a worker only after it
// observes the new `generation` (`Acquire`, pairing with the `Release` bump that
// follows the write) and before its `pending` decrement (`Release`). The
// pointer inside is only dereferenced under that protocol.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// `threads` shards: the caller of [`ShardPool::run`] runs shard 0 and
/// `threads - 1` persistent workers run the rest.
pub struct ShardPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl ShardPool {
    /// # Panics
    /// If `threads == 0` or a worker thread cannot be spawned.
    pub fn new(threads: usize) -> Self {
        assert!(threads > 0, "a shard pool needs at least one thread");
        let shared = Arc::new(Shared {
            generation: AtomicU64::new(0),
            pending: AtomicUsize::new(0),
            job: UnsafeCell::new(None),
            panicked: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
        });
        let workers = (1..threads)
            .map(|index| {
                let shared = Arc::clone(&shared);
                thread::Builder::new()
                    .name(format!("astrum-fdm-{index}"))
                    .spawn(move || worker(&shared, index))
                    .expect("spawning an FDM worker thread")
            })
            .collect();
        Self { shared, workers }
    }

    /// Number of shards (the caller's plus one per worker).
    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Run `f(i, &mut shards[i])` for every shard, shard `i` on worker `i`
    /// (shard 0 on the calling thread), and return once all have finished. A
    /// panic in any shard is re-raised here, after every shard has finished.
    ///
    /// # Panics
    /// If `shards.len() != self.threads()`, or a shard panicked.
    pub fn run<T: Send, F: Fn(usize, &mut T) + Sync>(&self, shards: &mut [T], f: F) {
        assert_eq!(shards.len(), self.threads(), "one shard per thread");
        struct Data<T, F> {
            shards: *mut T,
            f: *const F,
        }
        /// # Safety
        /// `data` points to a live `Data<T, F>` whose `shards` has an element
        /// `i` that no other thread is accessing.
        unsafe fn call<T, F: Fn(usize, &mut T)>(data: *const (), i: usize) {
            // SAFETY: guaranteed by the caller (see above).
            unsafe {
                let d = &*(data as *const Data<T, F>);
                (*d.f)(i, &mut *d.shards.add(i));
            }
        }
        let data = Data { shards: shards.as_mut_ptr(), f: &f as *const F };
        let job = Job { data: &data as *const Data<T, F> as *const (), call: call::<T, F> };
        let s = &*self.shared;
        // SAFETY: every worker is idle — the previous call returned only after
        // `pending` reached 0 — so nothing reads `job` while it is written.
        unsafe { *s.job.get() = Some(job) };
        s.pending.store(self.workers.len(), Ordering::Relaxed);
        s.generation.fetch_add(1, Ordering::Release);
        for w in &self.workers {
            w.thread().unpark();
        }
        // Shard 0 goes through the same raw pointer as the workers' shards, so
        // no reference to `shards` coexists with theirs. SAFETY: `data` lives
        // until this function returns; shard 0 is touched by no worker.
        let mine = panic::catch_unwind(AssertUnwindSafe(|| unsafe { call::<T, F>(job.data, 0) }));
        let waiting = Instant::now();
        while s.pending.load(Ordering::Acquire) != 0 {
            if waiting.elapsed() < SPIN {
                std::hint::spin_loop();
            } else {
                thread::yield_now();
            }
        }
        // SAFETY: every worker is done with this job.
        unsafe { *s.job.get() = None };
        if let Err(payload) = mine {
            panic::resume_unwind(payload);
        }
        if s.panicked.swap(false, Ordering::AcqRel) {
            panic!("a shard-pool worker panicked");
        }
    }
}

impl Drop for ShardPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        for w in &self.workers {
            w.thread().unpark();
        }
        for w in self.workers.drain(..) {
            // A worker never unwinds (shard panics are caught), so join cannot fail.
            let _ = w.join();
        }
    }
}

fn worker(s: &Shared, index: usize) {
    let mut seen = 0;
    loop {
        let idle = Instant::now();
        loop {
            let generation = s.generation.load(Ordering::Acquire);
            if generation != seen {
                seen = generation;
                break;
            }
            if s.shutdown.load(Ordering::Acquire) {
                return;
            }
            if idle.elapsed() < SPIN {
                std::hint::spin_loop();
            } else {
                // `run` unparks every worker after each bump, and park/unpark
                // keeps a token, so a bump between the check and here is not lost.
                thread::park();
            }
        }
        // SAFETY: published before this generation (see `Shared`) and valid
        // until this worker decrements `pending`.
        let job = unsafe { (*s.job.get()).expect("each generation publishes a job") };
        // SAFETY: `run` hands shard `index` to this worker only.
        let ran = panic::catch_unwind(AssertUnwindSafe(|| unsafe { (job.call)(job.data, index) }));
        if ran.is_err() {
            s.panicked.store(true, Ordering::Release);
        }
        s.pending.fetch_sub(1, Ordering::Release);
    }
}
