//! The bridge's threads end with their attachment: after many attach,
//! re-attach and detach cycles against a live stand-in sim, none is left.
//! (Its own test binary, so no other test's bridge threads are around.)

#![cfg(target_os = "linux")]

mod common;

use astrum_web::bridge::Bridge;
use common::{config, endpoint, wait_until, Sim};

/// Names (`comm`, at most 15 bytes) of this process's bridge threads: the
/// attachment threads (`bridge-run-<N>`) and their pumps.
fn bridge_threads() -> Vec<String> {
    std::fs::read_dir("/proc/self/task")
        .unwrap()
        .filter_map(|task| std::fs::read_to_string(task.ok()?.path().join("comm")).ok())
        .map(|comm| comm.trim_end().to_owned())
        .filter(|comm| comm.starts_with("bridge-"))
        .collect()
}

#[test]
fn attach_and_detach_cycles_leave_no_bridge_threads() {
    let at = endpoint(0);
    let _sim = Sim::flying(&at, 1);
    let bridge = Bridge::new(config());
    for cycle in 0..10 {
        let run = bridge.attach(&at);
        if cycle % 2 == 0 {
            // Let this attachment connect and start every pump...
            wait_until("the attachment to connect", || bridge.info().connected);
            let own = format!("bridge-run-{run}");
            wait_until("the attachment's threads", || {
                let threads = bridge_threads();
                threads.contains(&own)
                    && ["bridge-telemetr", "bridge-events", "bridge-models"]
                        .iter()
                        .all(|name| threads.iter().any(|t| t == name))
            });
        } // ...odd cycles are superseded while still connecting.
        if cycle % 3 == 0 {
            bridge.detach();
        }
    }
    bridge.detach();
    wait_until("every bridge thread to exit", || bridge_threads().is_empty());
}
