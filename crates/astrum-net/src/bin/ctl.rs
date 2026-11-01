//! `astrum-ctl`: talk to a running sim's control plane — status, pause, resume,
//! step, speed.

use std::process::ExitCode;
use std::time::Duration;

use astrum_net::{fetch_status, send_control, ControlRequest, RunStatus, Session};
use clap::{Parser, Subcommand};

/// Control a running astrum-sim over Zenoh.
#[derive(Parser)]
#[command(name = "astrum-ctl", version, about)]
struct Cli {
    /// The sim's Zenoh endpoint.
    #[arg(long, env = "ASTRUM_ZENOH_CONNECT", default_value_t = astrum_net::DEFAULT_CONNECT_ENDPOINT.to_string())]
    connect: String,

    /// Seconds to wait for the sim's reply.
    #[arg(long, default_value_t = 2.0)]
    timeout: f64,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print the sim's status (tick, state, speed, controllers).
    Status {
        /// Print the raw JSON.
        #[arg(long)]
        json: bool,
        /// Print it again every second until interrupted.
        #[arg(long)]
        watch: bool,
    },
    /// Pause after the current tick.
    Pause,
    /// Run on after a pause.
    Resume,
    /// Run N ticks, then pause.
    Step { ticks: u64 },
    /// Set the pace: × real time, 0 = as fast as possible.
    Speed { value: f64 },
}

fn show(s: &RunStatus, json: bool) -> String {
    if json {
        return serde_json::to_string(s).unwrap_or_default();
    }
    let pace = if s.speed > 0.0 { format!("{}×", s.speed) } else { "unpaced".into() };
    let mut line = format!(
        "{} | tick {} | sim {:.1} s | {:.1}× real time (target {pace}) | {} aircraft | controllers {:?}",
        s.state, s.tick, s.sim_time_s, s.achieved_speed, s.aircraft, s.controllers
    );
    if !s.lockstep_waiting.is_empty() {
        line += &format!(" | waiting on {:?}", s.lockstep_waiting);
    }
    line
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let timeout = Duration::from_secs_f64(cli.timeout);
    let session = match Session::connect(&cli.connect, timeout) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[ctl] no sim at {}: {e}", cli.connect);
            return ExitCode::FAILURE;
        }
    };
    let result = match cli.cmd {
        Cmd::Status { json, watch } => loop {
            match fetch_status(&session, timeout) {
                Ok(s) => println!("{}", show(&s, json)),
                Err(e) => break Err(e),
            }
            if !watch {
                break Ok(());
            }
            std::thread::sleep(Duration::from_secs(1));
        },
        Cmd::Pause => send_control(&session, ControlRequest::Pause, timeout),
        Cmd::Resume => send_control(&session, ControlRequest::Resume, timeout),
        Cmd::Step { ticks } => send_control(&session, ControlRequest::Step(ticks), timeout),
        Cmd::Speed { value } => send_control(&session, ControlRequest::Speed(value), timeout),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(astrum_net::NetError::Timeout) => {
            eprintln!("[ctl] no reply from a sim at {} within {:.1} s", cli.connect, cli.timeout);
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("[ctl] {e}");
            ExitCode::FAILURE
        }
    }
}
