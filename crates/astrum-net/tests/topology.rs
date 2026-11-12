//! The real star topology over TCP: a hub opened with `Session::listen` and two
//! connect-only clients. One client's publications must reach the other through
//! the hub (a recorder must see a controller's commands).

use std::time::{Duration, Instant};

use astrum_net::{zenoh, CommandSender, Session};
use astrum_proto::{decode_command, Control, RawActuators};
use zenoh::Wait;

#[test]
fn client_publications_are_routed_through_the_hub_to_other_clients() {
    // A port unlikely to collide with other tests or a running sim.
    let port = 20_000 + std::process::id() % 20_000;
    let hub = Session::listen(&format!("tcp/127.0.0.1:{port}")).unwrap();
    let connect = format!("tcp/127.0.0.1:{port}");
    let controller = Session::connect(&connect, Duration::from_secs(10)).unwrap();
    let recorder = Session::connect(&connect, Duration::from_secs(10)).unwrap();

    let sub = recorder.raw().declare_subscriber("astrum/command/*").wait().unwrap();
    let mut tx = CommandSender::new(&controller, 4).unwrap();
    let raw =
        Control::Raw(RawActuators { aileron: 0.25, elevator: 0.0, rudder: 0.0, throttle: 0.5 });

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut got = None;
    while got.is_none() && Instant::now() < deadline {
        tx.send(raw, false, 0).unwrap();
        if let Ok(Some(sample)) = sub.recv_timeout(Duration::from_millis(50)) {
            got = Some(decode_command(&sample.payload().to_bytes()).unwrap());
        }
    }
    drop(hub);
    let cmd = got.expect("the recorder never received the controller's command through the hub");
    assert_eq!(cmd.target_id, 4);
    assert_eq!(cmd.control, raw);
}
