//! The shard pool behind parallel FDM stepping: each shard runs exactly once per
//! call, always on the same thread (shard 0 on the caller), and a panicking
//! shard is reported without wedging the pool.

use std::panic::{self, AssertUnwindSafe};
use std::sync::Mutex;
use std::thread::{self, ThreadId};

use astrum_sim::step_pool::ShardPool;

#[test]
fn every_shard_runs_once_per_call_on_a_fixed_thread() {
    let pool = ShardPool::new(4);
    let caller = thread::current().id();
    let mut owners: Vec<Option<ThreadId>> = vec![None; 4];
    let mut counts = vec![0u64; 4];
    for round in 0..10_000u64 {
        let seen = Mutex::new(vec![None; 4]);
        pool.run(&mut counts, |i, count| {
            *count += 1;
            seen.lock().unwrap()[i] = Some(thread::current().id());
        });
        for (i, id) in seen.into_inner().unwrap().into_iter().enumerate() {
            let id = id.expect("every shard ran");
            assert_eq!(
                *owners[i].get_or_insert(id),
                id,
                "shard {i} moved threads in round {round}"
            );
        }
    }
    assert_eq!(counts, vec![10_000; 4]);
    assert_eq!(owners[0], Some(caller), "shard 0 runs on the caller");
    let distinct: std::collections::HashSet<_> = owners.iter().collect();
    assert_eq!(distinct.len(), 4, "one thread per shard");
}

#[test]
fn a_parked_pool_wakes_for_the_next_call() {
    // Longer than the spin window, so the workers park between calls.
    let pool = ShardPool::new(3);
    let mut shards = vec![0u32; 3];
    for _ in 0..5 {
        thread::sleep(astrum_sim::step_pool::SPIN * 20);
        pool.run(&mut shards, |_, s| *s += 1);
    }
    assert_eq!(shards, vec![5; 3]);
}

#[test]
fn a_panicking_shard_is_re_raised_and_the_pool_stays_usable() {
    let pool = ShardPool::new(3);
    let mut shards = vec![0u32; 3];
    for bad in [2, 0] {
        let hit = panic::catch_unwind(AssertUnwindSafe(|| {
            pool.run(&mut shards, |i, s| {
                *s += 1;
                assert_ne!(i, bad, "shard {bad} fails");
            })
        }));
        assert!(hit.is_err(), "the panic in shard {bad} reaches the caller");
    }
    // Every shard still ran both times, and the pool keeps working.
    assert_eq!(shards, vec![2; 3]);
    pool.run(&mut shards, |_, s| *s += 1);
    assert_eq!(shards, vec![3; 3]);
}
