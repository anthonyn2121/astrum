//! Live runs: a run's processes start together, their output arrives whole and
//! tagged, and the run's status follows how it ended — the sim's own exit, a
//! stop request, the run timeout, a failing controller (fail-fast or not) or a
//! companion that quits — with nothing left running afterwards; one run's stop,
//! kill or drop leaves the others alone; a sweep's process-mode run and
//! `astrum run` on the terminal fly through it. The "sim" is a shell script
//! standing in for `astrum-sim`, so no JSBSim is needed.

mod common;

use std::io::{BufRead, BufReader, Read};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, Receiver};
use std::sync::Mutex;
use std::time::Duration;

use astrum_sim::launch::{
    connect_endpoint, run_processes, Children, Companion, LiveRun, LiveSpec, Log, LogLine,
    LogStream, ProcessCtx, StartError,
};
use astrum_sim::scenario::Scenario;
use astrum_sim::sweep::{Outcome, Purpose, Status, Sweep, SweepFile};
use astrum_sim::systems::StopRequested;

/// Tests spawn processes from scripts they just wrote; one at a time, so no
/// other test's fork holds a script open for writing (ETXTBSY).
static SERIAL: Mutex<()> = Mutex::new(());

const STOPPED: &str =
    "echo '[astrum-sim] stopped at tick 300: sim 2.500 s in 0.100 s wall (25.0× real time)'";
/// A sim that runs until SIGTERM, then stops cleanly.
const UNTIL_TERM: &str = "trap 'echo stopping; exit 0' TERM\nwhile true; do sleep 0.02; done";
/// A sim that ignores SIGTERM (and so do the processes it starts); says
/// `ready` once it does.
const WEDGED: &str = "trap '' TERM\necho ready\nwhile true; do sleep 0.02; done";

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

fn companion(program: &str, args: &[&str], announce: Option<&str>) -> Companion {
    Companion {
        name: "the viewer".into(),
        source: "web".into(),
        program: program.into(),
        args: args.iter().map(Into::into).collect(),
        announce: announce.map(Into::into),
    }
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
    s.companions = vec![companion("sh", &["-c", "echo up; exit 1"], None)];
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
    s.companions = vec![companion("sleep", &["30"], None)];
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
    let Err(StartError::Launch(why)) = LiveRun::start(s, &children) else { panic!("started") };
    assert!(why.starts_with("controller 2: starting `/nonexistent/ctl`: "), "{why}");
    assert!(children.pids().is_empty());

    let mut s = spec(dir.path().join("sim"), channel().0);
    s.controllers = vec!["unbalanced 'quote".into()];
    let Err(StartError::Launch(why)) = LiveRun::start(s, &children) else { panic!("started") };
    assert_eq!(why, "controller 1: cannot split `unbalanced 'quote`");
}

#[test]
fn a_missing_sim_is_an_error_naming_it() {
    let _serial = serial();
    let s = spec("/nonexistent/astrum-sim".into(), channel().0);
    let Err(StartError::Setup(why)) = LiveRun::start(s, &Children::new()) else {
        panic!("started")
    };
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
    stop.0.store(true, Ordering::Relaxed);
    for live in runs {
        assert_eq!(live.wait().status, Status::Stopped("stop requested".into()));
    }
    assert!(children.pids().is_empty());
}

#[test]
fn stopping_dropping_or_killing_one_run_leaves_the_others_sharing_its_handles() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let sim = script(dir.path(), "sim", WEDGED);
    let children = Children::new();
    let stop = StopRequested::default();
    let start = || {
        let mut s = spec(sim.clone(), channel().0);
        s.stop = stop.clone();
        LiveRun::start(s, &children).unwrap()
    };
    let (stopped, dropped, killed, mut survivor) = (start(), start(), start(), start());
    stopped.stop();
    killed.kill();
    let killed = killed.wait();
    assert!(
        matches!(&killed.status, Status::Failed(why) if why.starts_with("sim exited (signal: 9")),
        "{killed:?}"
    );
    std::thread::spawn(move || drop(dropped)).join().unwrap();
    assert!(!stop.requested(), "a run set the shared stop");
    assert!(survivor.poll().is_none(), "the survivor ended");
    assert_eq!(stopped.wait().status, Status::Stopped("stop requested".into()));
    assert!(survivor.poll().is_none(), "the survivor ended");
    survivor.kill();
    survivor.wait();
    assert!(children.pids().is_empty());
}

#[test]
fn a_stopped_sim_that_exits_non_zero_is_still_stopped() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let sim = script(dir.path(), "sim", "trap 'exit 1' TERM\nwhile true; do sleep 0.02; done");
    let live = LiveRun::start(spec(sim, channel().0), &Children::new()).unwrap();
    std::thread::sleep(Duration::from_millis(100));
    live.stop();
    assert_eq!(live.wait().status, Status::Stopped("stop requested".into()));
}

#[test]
fn waiting_on_a_stopped_run_kills_a_sim_that_ignores_the_stop_after_the_grace_period() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let sim = script(dir.path(), "sim", WEDGED);
    let children = Children::new();
    let (log, rx) = channel();
    let live = LiveRun::start(spec(sim, log), &children).unwrap();
    assert_eq!(rx.recv().unwrap(), line("sim", LogStream::Stdout, "ready"));
    live.stop();
    let outcome = live.wait();
    assert_eq!(outcome.status, Status::Stopped("stop requested".into()));
    assert!((4.5..10.0).contains(&outcome.wall_s), "killed after {} s", outcome.wall_s);
    assert!(children.pids().is_empty());
}

#[test]
fn a_background_grandchild_holding_the_output_does_not_hold_up_the_run() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let sim = script(dir.path(), "sim", "sleep 0.3");
    let (log, rx) = channel();
    let mut s = spec(sim, log);
    s.controllers = vec!["sh -c 'sleep 6 & echo started'".into()];
    let outcome = LiveRun::start(s, &Children::new()).unwrap().wait();
    assert_eq!(outcome.status, Status::Ok);
    assert!(outcome.wall_s < 3.0, "waited {} s for the grandchild", outcome.wall_s);
    assert!(rx.try_iter().any(|l| l == line("ctl1", LogStream::Stdout, "started")));
}

#[test]
fn a_companion_is_announced_once_started_before_the_controllers() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let sim = script(dir.path(), "sim", "sleep 0.3");
    let (log, rx) = channel();
    let mut s = spec(sim, log);
    s.companions = vec![companion("sleep", &["30"], Some("viewer: http://localhost:8080"))];
    s.controllers = vec!["echo ctl".into()];
    assert_eq!(LiveRun::start(s, &Children::new()).unwrap().wait().status, Status::Ok);
    let lines: Vec<LogLine> = rx.try_iter().collect();
    let at = |l: LogLine| lines.iter().position(|x| *x == l);
    let announced = at(line("astrum", LogStream::Stdout, "viewer: http://localhost:8080"));
    let ctl = at(line("ctl1", LogStream::Stdout, "ctl"));
    assert!(announced.is_some() && announced < ctl, "{lines:?}");
}

#[test]
fn wildcard_listen_addresses_connect_over_loopback() {
    assert_eq!(connect_endpoint("tcp/0.0.0.0:7447"), "tcp/127.0.0.1:7447");
    assert_eq!(connect_endpoint("tcp/[::]:7447"), "tcp/[::1]:7447");
    assert_eq!(connect_endpoint("tcp/10.0.0.2:7447"), "tcp/10.0.0.2:7447");
}

#[test]
fn outcomes_and_log_lines_serialise_for_the_web_api() {
    let outcome =
        Outcome { status: Status::Failed("sim exited".into()), sim_time_s: Some(2.5), wall_s: 1.0 };
    assert_eq!(
        serde_json::to_value(&outcome).unwrap(),
        serde_json::json!({
            "status": {"kind": "failed", "reason": "sim exited"},
            "sim_time_s": 2.5,
            "wall_s": 1.0,
        })
    );
    assert_eq!(serde_json::to_value(Status::Ok).unwrap(), serde_json::json!({"kind": "ok"}));
    assert_eq!(
        serde_json::to_value(Status::Stopped("stop requested".into())).unwrap(),
        serde_json::json!({"kind": "stopped", "reason": "stop requested"})
    );
    assert_eq!(
        serde_json::to_value(line("ctl1", LogStream::Stderr, "x")).unwrap(),
        serde_json::json!({"source": "ctl1", "stream": "stderr", "line": "x"})
    );
}

#[test]
fn a_process_mode_sweep_run_listens_on_its_port_records_and_keeps_its_scenario() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let mut scenario = common::single_ship();
    scenario.sim.max_time_s = 1.0;
    let base = dir.path().join("base.toml");
    std::fs::write(&base, scenario.to_toml()).unwrap();
    let mut file = SweepFile::for_scenario(base);
    file.repeats = 2;
    file.speed = Some(1.0);
    file.controllers = vec!["echo ctl {endpoint}".into()];
    let sweep = Sweep::plan(file, "t", Purpose::Sweep).unwrap();
    let out = dir.path().join("out");
    std::fs::create_dir(&out).unwrap();
    let sim = script(dir.path(), "sim", &format!("echo \"args $@\"\nsleep 0.3\n{STOPPED}"));
    let stop = StopRequested::default();
    let children = Children::new();
    let ctx = ProcessCtx {
        sweep: &sweep,
        out: &out,
        sim: &sim,
        base_port: 27610,
        stop: &stop,
        children: &children,
    };
    let outcome = run_processes(&ctx, &sweep.runs[1]);
    assert_eq!(outcome.status, Status::Ok);
    assert_eq!(outcome.sim_time_s, Some(2.5));
    let read = |name: &str| std::fs::read_to_string(out.join(name)).unwrap();
    let o = out.display();
    assert!(
        read("run-0002.sim.log").starts_with(&format!(
            "args {o}/run-0002.toml --listen tcp/127.0.0.1:27611 --record {o}/run-0002.mcap\n"
        )),
        "{}",
        read("run-0002.sim.log")
    );
    assert_eq!(read("run-0002.ctl1.log"), "ctl tcp/127.0.0.1:27611\n");
    assert_eq!(Scenario::load(out.join("run-0002.toml")).unwrap(), sweep.runs[1].scenario);
}

// --- `astrum run` on the terminal ---------------------------------------------

/// `astrum run <file> --sim <sim>` on a port of its own.
fn astrum_run(file: &Path, sim: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_astrum"));
    cmd.arg("run").arg(file).arg("--sim").arg(sim).args(["--listen", "tcp/127.0.0.1:27600"]);
    cmd
}

fn scenario_file(dir: &Path) -> PathBuf {
    let path = dir.join("scenario.cli.toml");
    std::fs::write(&path, common::single_ship().to_toml()).unwrap();
    path
}

#[test]
fn astrum_run_tags_each_line_without_colour_when_piped_and_exits_0_when_ok() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let sim = script(dir.path(), "sim", "echo hello\necho warn >&2");
    let file = scenario_file(dir.path());
    let out = astrum_run(&file, &sim).output().unwrap();
    assert_eq!(out.status.code(), Some(0));
    let f = file.display();
    assert_eq!(String::from_utf8_lossy(&out.stdout), format!("[sim]  hello\n[astrum] ok: {f}\n"));
    assert_eq!(String::from_utf8_lossy(&out.stderr), "[sim]  warn\n");
}

#[test]
fn astrum_run_exits_1_when_the_sim_fails_or_cannot_start() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let file = scenario_file(dir.path());
    let f = file.display();

    let out = astrum_run(&file, &script(dir.path(), "sim", "exit 3")).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    let expected = format!("[astrum] FAILED: {f}: sim exited (exit status: 3)\n");
    assert_eq!(String::from_utf8_lossy(&out.stderr), expected);

    let not_executable = dir.path().join("not-executable");
    std::fs::write(&not_executable, "").unwrap();
    let out = astrum_run(&file, &not_executable).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    let expected = format!(
        "[astrum] starting {}: Permission denied (os error 13)\n",
        not_executable.display()
    );
    assert_eq!(String::from_utf8_lossy(&out.stderr), expected);

    let run = dir.path().join("run.toml");
    let base = format!("base = \"{f}\"\nspeed = 1.0\ncontrollers = [\"/nonexistent/ctl\"]\n");
    std::fs::write(&run, base).unwrap();
    let out = astrum_run(&run, &script(dir.path(), "sim", UNTIL_TERM)).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    let expected = format!(
        "[astrum] FAILED: {}: controller 1: starting `/nonexistent/ctl`: No such file or \
         directory (os error 2)\n",
        run.display()
    );
    assert!(String::from_utf8_lossy(&out.stderr).ends_with(&expected), "{out:?}");
}

#[test]
fn astrum_run_exits_2_on_an_invalid_file() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("bad.toml");
    std::fs::write(&file, "[sim]\nrate_hz = \n").unwrap();
    let out = astrum_run(&file, Path::new("/nonexistent/astrum-sim")).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn astrum_run_stopped_by_ctrl_c_stops_the_sim_cleanly_and_exits_130() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let sim = script(
        dir.path(),
        "sim",
        // `wait`, not a foreground `sleep`: sh reports a foreground child the
        // stop's SIGTERM kills ("Terminated").
        "trap 'echo stopping; exit 0' TERM\necho ready\nsleep 30 &\nwait",
    );
    let file = scenario_file(dir.path());
    let mut child =
        astrum_run(&file, &sim).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut first = String::new();
    stdout.read_line(&mut first).unwrap();
    assert_eq!(first, "[sim]  ready\n");
    // SAFETY: SIGINT to the child we spawned and have not reaped.
    unsafe {
        libc::kill(child.id() as libc::pid_t, libc::SIGINT);
    }
    let mut rest = String::new();
    stdout.read_to_string(&mut rest).unwrap();
    let mut stderr = String::new();
    child.stderr.take().unwrap().read_to_string(&mut stderr).unwrap();
    assert_eq!(child.wait().unwrap().code(), Some(130));
    assert_eq!(rest, "[sim]  stopping\n");
    let f = file.display();
    assert_eq!(
        stderr,
        format!(
            "[astrum] stopping the run (Ctrl-C again to kill it)\n\
             [astrum] stopped: {f}: stop requested\n"
        )
    );
}
