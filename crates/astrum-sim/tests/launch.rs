//! Live runs: a run's processes start together, their output arrives whole and
//! tagged, and the run's status follows how it ended — the sim's own exit, a
//! stop request, the run timeout, a failing controller (fail-fast or not) or a
//! companion that quits — with nothing left running afterwards. The "sim" is a
//! shell script standing in for `astrum-sim`, so no JSBSim is needed.

mod common;

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::sync::Mutex;
use std::time::Duration;

use astrum_sim::launch::{Children, Companion, LiveRun, LiveSpec, Log, LogLine, LogStream};
use astrum_sim::sweep::Status;
use astrum_sim::systems::StopRequested;

/// Tests spawn processes from scripts they just wrote; one at a time, so no
/// other test's fork holds a script open for writing (ETXTBSY).
static SERIAL: Mutex<()> = Mutex::new(());

const STOPPED: &str =
    "echo '[astrum-sim] stopped at tick 300: sim 2.500 s in 0.100 s wall (25.0× real time)'";
/// A sim that runs until SIGTERM, then stops cleanly.
const UNTIL_TERM: &str = "trap 'echo stopping; exit 0' TERM\nwhile true; do sleep 0.02; done";

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn spec(sim: PathBuf, log: Log) -> LiveSpec {
    LiveSpec {
        sim,
        scenario: common::single_ship(),
        scenario_file: None,
        listen: "tcp/0.0.0.0:27600".into(),
        record: None,
        companions: Vec::new(),
        controllers: Vec::new(),
        run_timeout: None,
        fail_fast: true,
        log,
        stop: StopRequested::default(),
    }
}

fn channel() -> (Log, Receiver<LogLine>) {
    let (tx, rx) = mpsc::channel();
    (Log::Channel(tx), rx)
}

fn line(source: &str, stream: LogStream, text: &str) -> LogLine {
    LogLine { source: source.into(), stream, line: text.into() }
}

#[test]
fn output_arrives_as_whole_tagged_lines_and_a_clean_sim_exit_is_ok() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let sim = script(
        dir.path(),
        "sim",
        &format!(
            "grep -q '^rate_hz' \"$1\" && echo \"scenario $2 $3\"\nprintf 'partial'; echo ' line'\n\
             echo oops >&2\nsleep 0.3\n{STOPPED}"
        ),
    );
    let (log, rx) = channel();
    let mut s = spec(sim, log);
    s.controllers = vec!["sh -c 'echo \"ctl at $ASTRUM_ZENOH_CONNECT\"'".into()];
    let children = Children::new();
    let live = LiveRun::start(s, &children).unwrap();
    assert_eq!(live.endpoint(), "tcp/127.0.0.1:27600");
    let outcome = live.wait();
    assert_eq!(outcome.status, Status::Ok);
    assert_eq!(outcome.sim_time_s, Some(2.5));
    assert!(children.pids().is_empty());
    let lines: Vec<LogLine> = rx.try_iter().collect();
    for expected in [
        line("sim", LogStream::Stdout, "scenario --listen tcp/0.0.0.0:27600"),
        line("sim", LogStream::Stdout, "partial line"),
        line("sim", LogStream::Stderr, "oops"),
        line("ctl1", LogStream::Stdout, "ctl at tcp/127.0.0.1:27600"),
    ] {
        assert!(lines.contains(&expected), "{expected:?} not in {lines:?}");
    }
}

#[test]
fn a_stop_request_stops_the_sim_cleanly_and_removes_the_temporary_scenario() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let sim = script(dir.path(), "sim", &format!("echo \"file $1\"\n{UNTIL_TERM}"));
    let (log, rx) = channel();
    let mut live = LiveRun::start(spec(sim, log), &Children::new()).unwrap();
    std::thread::sleep(Duration::from_millis(200));
    assert!(live.poll().is_none(), "the run ended by itself");
    live.stop();
    let outcome = live.wait();
    assert_eq!(outcome.status, Status::Stopped("stop requested".into()));
    let lines: Vec<LogLine> = rx.try_iter().collect();
    let file = lines.iter().find_map(|l| l.line.strip_prefix("file ")).expect("the sim's file");
    assert!(!Path::new(file).exists(), "{file} outlived the run");
    assert!(lines.contains(&line("sim", LogStream::Stdout, "stopping")), "{lines:?}");
}

#[test]
fn poll_reports_the_outcome_once_the_run_has_ended() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let sim = script(dir.path(), "sim", "exit 0");
    let mut live = LiveRun::start(spec(sim, channel().0), &Children::new()).unwrap();
    let mut polls = 0;
    while live.poll().is_none() {
        polls += 1;
        assert!(polls < 500, "the run never ended");
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(live.poll().map(|o| o.status.clone()), Some(Status::Ok));
    assert_eq!(live.wait().status, Status::Ok);
}

#[test]
fn the_run_timeout_stops_the_run() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let sim = script(dir.path(), "sim", UNTIL_TERM);
    let mut s = spec(sim, channel().0);
    s.run_timeout = Some(Duration::from_millis(200));
    let outcome = LiveRun::start(s, &Children::new()).unwrap().wait();
    assert_eq!(outcome.status, Status::Stopped("run_timeout_s 0.2 exceeded".into()));
}

#[test]
fn a_failing_controller_fails_the_run_and_fail_fast_stops_it() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let sim = script(dir.path(), "sim", UNTIL_TERM);
    let mut s = spec(sim, channel().0);
    s.controllers = vec!["sh -c 'exit 0'".into(), "sh -c 'exit 3'".into()];
    let outcome = LiveRun::start(s, &Children::new()).unwrap().wait();
    assert_eq!(outcome.status, Status::Failed("controller 2 exited (exit status: 3)".into()));
}

#[test]
fn without_fail_fast_the_run_flies_on_and_fails_at_its_end() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let sim = script(dir.path(), "sim", "sleep 0.3\necho done");
    let (log, rx) = channel();
    let mut s = spec(sim, log);
    s.controllers = vec!["sh -c 'exit 3'".into()];
    s.fail_fast = false;
    let outcome = LiveRun::start(s, &Children::new()).unwrap().wait();
    assert_eq!(outcome.status, Status::Failed("controller 1 exited (exit status: 3)".into()));
    assert!(rx.try_iter().any(|l| l == line("sim", LogStream::Stdout, "done")));
}

#[test]
fn a_sim_that_fails_by_itself_is_the_problem_even_after_a_controller_quit() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let sim = script(dir.path(), "sim", "sleep 0.2\nexit 2");
    let mut s = spec(sim, channel().0);
    s.controllers = vec!["sh -c 'exit 1'".into()];
    s.fail_fast = false;
    let outcome = LiveRun::start(s, &Children::new()).unwrap().wait();
    assert_eq!(outcome.status, Status::Failed("sim exited (exit status: 2)".into()));
}

#[test]
fn a_companion_that_exits_is_reported_and_the_run_goes_on() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let sim = script(dir.path(), "sim", "sleep 0.3");
    let (log, rx) = channel();
    let mut s = spec(sim, log);
    s.companions = vec![Companion {
        name: "the viewer".into(),
        source: "web".into(),
        program: "sh".into(),
        args: vec!["-c".into(), "echo up; exit 1".into()],
    }];
    let outcome = LiveRun::start(s, &Children::new()).unwrap().wait();
    assert_eq!(outcome.status, Status::Ok);
    let lines: Vec<LogLine> = rx.try_iter().collect();
    assert!(lines.contains(&line("web", LogStream::Stdout, "up")), "{lines:?}");
    let notice =
        line("astrum", LogStream::Stderr, "the viewer exited (exit status: 1); the run goes on");
    assert!(lines.contains(&notice), "{lines:?}");
}

#[test]
fn companions_and_controllers_are_stopped_with_the_run() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let sim = script(dir.path(), "sim", "sleep 0.2");
    let mut s = spec(sim, channel().0);
    s.companions = vec![Companion {
        name: "the viewer".into(),
        source: "web".into(),
        program: "sleep".into(),
        args: vec!["30".into()],
    }];
    s.controllers = vec!["sleep 30".into()];
    let children = Children::new();
    let outcome = LiveRun::start(s, &children).unwrap().wait();
    assert_eq!(outcome.status, Status::Ok);
    assert!(outcome.wall_s < 10.0, "waited out the companions: {} s", outcome.wall_s);
    assert!(children.pids().is_empty());
}

#[test]
fn a_controller_that_cannot_start_is_an_error_and_leaves_nothing_running() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let sim = script(dir.path(), "sim", UNTIL_TERM);
    let mut s = spec(sim, channel().0);
    s.controllers = vec!["sh -c 'sleep 30'".into(), "/nonexistent/ctl --id 2".into()];
    let children = Children::new();
    let why = LiveRun::start(s, &children).err().expect("the run started");
    assert!(why.starts_with("controller 2: starting `/nonexistent/ctl`: "), "{why}");
    assert!(children.pids().is_empty());

    let mut s = spec(dir.path().join("sim"), channel().0);
    s.controllers = vec!["unbalanced 'quote".into()];
    let why = LiveRun::start(s, &children).err().expect("the run started");
    assert_eq!(why, "controller 1: cannot split `unbalanced 'quote`");
}

#[test]
fn a_missing_sim_is_an_error_naming_it() {
    let _serial = serial();
    let s = spec("/nonexistent/astrum-sim".into(), channel().0);
    let why = LiveRun::start(s, &Children::new()).err().expect("the run started");
    assert!(why.starts_with("starting /nonexistent/astrum-sim: "), "{why}");
}

#[test]
fn dropping_a_running_run_stops_it_and_reaps_its_processes() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let sim = script(dir.path(), "sim", UNTIL_TERM);
    let mut s = spec(sim, channel().0);
    s.controllers = vec!["sleep 30".into()];
    let children = Children::new();
    let live = LiveRun::start(s, &children).unwrap();
    assert_eq!(children.pids().len(), 2);
    drop(live);
    assert!(children.pids().is_empty());
}

#[test]
fn file_logs_hold_each_process_and_point_at_the_sim_log_on_failure() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    // The sim outlives the controller: a sim's exit stops what still runs.
    let sim = script(dir.path(), "sim", &format!("sleep 0.3\n{STOPPED}\nexit 1"));
    let mut s = spec(sim, Log::Files { dir: dir.path().to_owned(), stem: "run-0007".into() });
    s.scenario_file = Some(dir.path().join("run-0007.toml"));
    s.controllers = vec!["echo hello".into()];
    s.fail_fast = false;
    let outcome = LiveRun::start(s, &Children::new()).unwrap().wait();
    assert_eq!(
        outcome.status,
        Status::Failed("sim exited (exit status: 1); see run-0007.sim.log".into())
    );
    assert_eq!(outcome.sim_time_s, Some(2.5));
    let read = |name: &str| std::fs::read_to_string(dir.path().join(name)).unwrap();
    assert!(read("run-0007.sim.log").contains("stopped at tick 300"));
    assert_eq!(read("run-0007.ctl1.log"), "hello\n");
    assert!(read("run-0007.toml").contains("rate_hz"), "the scenario file is kept");
}

#[test]
fn runs_fly_side_by_side_and_a_shared_stop_handle_stops_them_all() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let sim = script(dir.path(), "sim", UNTIL_TERM);
    let children = Children::new();
    let stop = StopRequested::default();
    let runs: Vec<LiveRun> = (0..2)
        .map(|k| {
            let mut s = spec(sim.clone(), channel().0);
            s.listen = format!("tcp/127.0.0.1:{}", 27601 + k);
            s.stop = stop.clone();
            LiveRun::start(s, &children).unwrap()
        })
        .collect();
    assert_eq!(children.pids().len(), 2);
    runs[0].stop();
    for live in runs {
        assert_eq!(live.wait().status, Status::Stopped("stop requested".into()));
    }
    assert!(children.pids().is_empty());
}
