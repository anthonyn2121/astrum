//! Initial-condition offsets: forward/right/up land in the right ENU
//! directions for any heading, headings wrap, climb and bank stay strictly
//! within ±90, a rigid group keeps its geometry, and random spreads stay in
//! bounds and reproduce per stream.

use astrum_geo::{Enu, Geodetic, LocalFrame};
use astrum_sim::perturb::{keys, wrap_heading, Axis, Group, Offset, Perturb, Spread};
use astrum_sim::rng::SplitMix64;
use astrum_sim::scenario::Scenario;

const TWO_SHIP: &str = r#"
[world]
origin_lat_deg = 37.6188
origin_lon_deg = -122.3756
origin_alt_hae_m = 0.0

[sim]
rate_hz = 120.0

[[aircraft]]
id = 1
model = "f16"
lat_deg = 37.6188
lon_deg = -122.3756
alt_hae_m = 4572.0
heading_deg = 90.0
tas_mps = 250.0

[[aircraft]]
id = 2
model = "f16"
lat_deg = 37.6200
lon_deg = -122.3700
alt_hae_m = 4877.0
heading_deg = 180.0
tas_mps = 260.0

[[aircraft]]
id = 3
model = "c172x"
lat_deg = 37.6100
lon_deg = -122.3800
alt_hae_m = 1500.0
heading_deg = 10.0
tas_mps = 55.0
"#;

fn scenario() -> Scenario {
    Scenario::from_toml_str(TWO_SHIP).unwrap()
}

fn offset(values: &[(Axis, f64)]) -> Offset {
    let mut o = Offset::default();
    for &(axis, v) in values {
        o.set(axis, v);
    }
    o
}

fn single(id: u32, values: &[(Axis, f64)]) -> Group {
    Group { lead: id, members: vec![id], offset: offset(values) }
}

fn geodetic(s: &Scenario, id: u32) -> Geodetic {
    let a = s.aircraft.iter().find(|a| a.id == id).unwrap();
    Geodetic::from_degrees(a.lat_deg, a.lon_deg, a.alt_hae_m)
}

/// Where aircraft `id` of `after` sits in a local ENU frame at its place in `before`.
fn moved(before: &Scenario, after: &Scenario, id: u32) -> Enu {
    LocalFrame::new(geodetic(before, id)).geodetic_to_enu(&geodetic(after, id))
}

fn distance(s: &Scenario, a: u32, b: u32) -> f64 {
    let frame = LocalFrame::new(geodetic(s, 1));
    let (p, q) = (frame.geodetic_to_enu(&geodetic(s, a)), frame.geodetic_to_enu(&geodetic(s, b)));
    ((p.e - q.e).powi(2) + (p.n - q.n).powi(2) + (p.u - q.u).powi(2)).sqrt()
}

fn close(got: Enu, want: (f64, f64, f64)) {
    let ok = (got.e - want.0).abs() < 1e-3
        && (got.n - want.1).abs() < 1e-3
        && (got.u - want.2).abs() < 1e-3;
    assert!(ok, "got {got:?}, want e/n/u {want:?}");
}

#[test]
fn forward_and_right_follow_the_heading_and_up_raises_the_aircraft() {
    // (heading, ENU of forward 100 m, ENU of right 50 m)
    for (heading, forward, right) in [
        (0.0, (0.0, 100.0), (50.0, 0.0)),
        (90.0, (100.0, 0.0), (0.0, -50.0)),
        (270.0, (-100.0, 0.0), (0.0, 50.0)),
    ] {
        let mut base = scenario();
        base.aircraft[0].heading_deg = heading;
        for (axis, want) in [(Axis::ForwardM, forward), (Axis::RightM, right)] {
            let v = if axis == Axis::ForwardM { 100.0 } else { 50.0 };
            let mut s = base.clone();
            single(1, &[(axis, v)]).apply(&mut s).unwrap();
            close(moved(&base, &s, 1), (want.0, want.1, 0.0));
            assert_eq!(s.aircraft[0].heading_deg, heading, "a shift does not turn");
        }
    }
    let base = scenario();
    let mut s = base.clone();
    single(1, &[(Axis::UpM, 30.0)]).apply(&mut s).unwrap();
    close(moved(&base, &s, 1), (0.0, 0.0, 30.0));
    assert!((s.aircraft[0].alt_hae_m - (base.aircraft[0].alt_hae_m + 30.0)).abs() < 1e-3);
    assert_eq!(s.aircraft[1], base.aircraft[1], "other aircraft stay put");
}

#[test]
fn a_shift_moves_the_aircraft_by_its_level_frame_length() {
    for heading in [0.0, 33.0, 90.0, 200.0, 359.0] {
        let mut base = scenario();
        base.aircraft[0].heading_deg = heading;
        let mut s = base.clone();
        single(1, &[(Axis::ForwardM, 3000.0), (Axis::RightM, -4000.0)]).apply(&mut s).unwrap();
        let d = moved(&base, &s, 1);
        let len = (d.e * d.e + d.n * d.n + d.u * d.u).sqrt();
        assert!((len - 5000.0).abs() < 1e-3, "heading {heading}: {len}");
        assert!(d.u.abs() < 1e-3, "level: {d:?}");
    }
}

#[test]
fn heading_offsets_wrap_and_climb_or_bank_stay_within_90() {
    let mut s = scenario();
    single(1, &[(Axis::HeadingDeg, 300.0)]).apply(&mut s).unwrap();
    assert_eq!(s.aircraft[0].heading_deg, 30.0);
    single(1, &[(Axis::HeadingDeg, -45.0)]).apply(&mut s).unwrap();
    assert_eq!(s.aircraft[0].heading_deg, 345.0);
    assert_eq!(wrap_heading(-1e-17), 0.0);
    assert_eq!(wrap_heading(720.0), 0.0);

    single(1, &[(Axis::ClimbDeg, 89.9), (Axis::BankDeg, -89.9)]).apply(&mut s).unwrap();
    for (axis, v, field) in
        [(Axis::ClimbDeg, 90.0, "climb_deg"), (Axis::BankDeg, -120.0, "bank_deg")]
    {
        let mut s = scenario();
        let e = single(2, &[(axis, v)]).apply(&mut s).unwrap_err();
        let key = format!("aircraft.2.offset.{field}");
        assert!(e.contains("aircraft 2") && e.contains(&key) && e.contains("(-90, 90)"), "{e}");
    }
}

#[test]
fn a_rigid_group_keeps_its_geometry_and_turns_about_its_lead() {
    let base = scenario();
    let group = Group {
        lead: 1,
        members: vec![1, 2, 3],
        offset: offset(&[
            (Axis::ForwardM, 200.0),
            (Axis::RightM, -100.0),
            (Axis::UpM, 50.0),
            (Axis::HeadingDeg, 40.0),
            (Axis::BankDeg, 5.0),
        ]),
    };
    let mut s = base.clone();
    group.apply(&mut s).unwrap();
    for (a, b) in [(1, 2), (1, 3), (2, 3)] {
        let (before, after) = (distance(&base, a, b), distance(&s, a, b));
        assert!((before - after).abs() < 1e-3, "{a}-{b}: {before} → {after}");
    }
    // The lead moves exactly as alone; everyone turns and banks alike.
    let mut alone = base.clone();
    Group { members: vec![1], ..group.clone() }.apply(&mut alone).unwrap();
    close(moved(&base, &s, 1), {
        let d = moved(&base, &alone, 1);
        (d.e, d.n, d.u)
    });
    for (a, b) in s.aircraft.iter().zip(&base.aircraft) {
        assert!((a.heading_deg - wrap_heading(b.heading_deg + 40.0)).abs() < 1e-9);
        assert_eq!(a.bank_deg, b.bank_deg + 5.0);
    }
    // Member 2 sits where the lead's frame, turned 40° clockwise, puts it.
    let frame = LocalFrame::new(geodetic(&base, 1));
    let p = frame.geodetic_to_enu(&geodetic(&base, 2));
    let q = frame.geodetic_to_enu(&geodetic(&s, 2));
    let lead = frame.geodetic_to_enu(&geodetic(&s, 1));
    let (sin, cos) = 40f64.to_radians().sin_cos();
    let want = (p.e * cos + p.n * sin, p.n * cos - p.e * sin, p.u);
    close(Enu { e: q.e - lead.e, n: q.n - lead.n, u: q.u - lead.u }, want);
}

#[test]
fn spreads_stay_in_bounds_and_reproduce_per_stream() {
    for spread in [Spread::Uniform(5.0), Spread::Normal(2.0)] {
        let mut rng = SplitMix64::stream(11, 0);
        let draws: Vec<f64> = (0..20_000).map(|_| spread.draw(&mut rng)).collect();
        let bound = spread.bound();
        assert!(draws.iter().all(|d| d.abs() <= bound), "{spread:?}");
        assert!(draws.iter().any(|d| d.abs() > 0.9 * bound), "{spread:?} uses its range");
        assert!(draws.iter().all(|d| (d * 1000.0 - (d * 1000.0).round()).abs() < 1e-6));
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        assert!(mean.abs() < 0.1, "{spread:?}: mean {mean}");
    }
    let stream = |seed, k| {
        let mut rng = SplitMix64::stream(seed, k);
        (0..8).map(|_| rng.next_u64()).collect::<Vec<_>>()
    };
    assert_eq!(stream(3, 5), stream(3, 5));
    assert_ne!(stream(3, 5), stream(3, 6));
    assert_ne!(stream(3, 5), stream(4, 5));
    // Neighbouring streams are not one stream shifted by a draw.
    assert_ne!(stream(3, 5)[1..], stream(3, 6)[..7]);
    assert_ne!(stream(3, 6)[1..], stream(3, 5)[..7]);
}

#[test]
fn draws_are_recorded_per_aircraft_or_under_a_rigid_groups_lead() {
    let table = |rigid| Perturb {
        aircraft: vec![2, 3],
        rigid,
        forward_m: Some(Spread::Uniform(10.0)),
        right_m: None,
        up_m: None,
        heading_deg: Some(Spread::Normal(1.0)),
        climb_deg: None,
        bank_deg: None,
    };
    assert_eq!(
        keys(&[table(false)]),
        [
            "aircraft.2.offset.forward_m",
            "aircraft.2.offset.heading_deg",
            "aircraft.3.offset.forward_m",
            "aircraft.3.offset.heading_deg"
        ]
    );
    assert_eq!(
        keys(&[table(true)]),
        ["aircraft.2.offset.forward_m", "aircraft.2.offset.heading_deg"]
    );
}
