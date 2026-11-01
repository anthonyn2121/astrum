//! Coordinate-conversion integration tests: WGS84 geodetic ↔ ECEF ↔ local-tangent
//! ENU round-trips, sign/scale checks and local-level ↔ grid vector rotation.

use astrum_geo::{ecef_to_geodetic, geodetic_to_ecef, Enu, Geodetic, LocalFrame, Ned};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn ecef_roundtrip() {
    let g = Geodetic::from_degrees(37.6188, -122.3756, 152.0);
    let g2 = ecef_to_geodetic(&geodetic_to_ecef(&g));
    assert!(close(g.lat_deg(), g2.lat_deg(), 1e-9), "{} vs {}", g.lat_deg(), g2.lat_deg());
    assert!(close(g.lon_deg(), g2.lon_deg(), 1e-9));
    assert!(close(g.alt_hae_m, g2.alt_hae_m, 1e-4), "{} vs {}", g.alt_hae_m, g2.alt_hae_m);
}

#[test]
fn ecef_roundtrip_is_exact_near_the_poles() {
    // The textbook `p / cos(lat)` height formula blows up here.
    for lat in [89.9999, 90.0, -90.0, -89.99] {
        let g = Geodetic::from_degrees(lat, 12.0, 10_000.0);
        let g2 = ecef_to_geodetic(&geodetic_to_ecef(&g));
        assert!(close(g.lat_deg(), g2.lat_deg(), 1e-9), "lat {lat}: {}", g2.lat_deg());
        assert!(close(g.alt_hae_m, g2.alt_hae_m, 1e-4), "lat {lat}: alt {}", g2.alt_hae_m);
    }
}

#[test]
fn origin_maps_to_zero() {
    let o = Geodetic::from_degrees(40.0, -105.0, 1600.0);
    let f = LocalFrame::new(o);
    let z = f.geodetic_to_enu(&o);
    assert!(close(z.e, 0.0, 1e-6) && close(z.n, 0.0, 1e-6) && close(z.u, 0.0, 1e-6), "{z:?}");
}

#[test]
fn north_east_up_signs_and_scale() {
    let o = Geodetic::from_degrees(40.0, -105.0, 1600.0);
    let f = LocalFrame::new(o);

    // ~0.001° north of origin: ~111 m North, ~0 East/Up.
    let north = f.geodetic_to_enu(&Geodetic::from_degrees(40.001, -105.0, 1600.0));
    assert!(north.n > 0.0, "north should be +N, got {}", north.n);
    assert!(close(north.e, 0.0, 1.0) && close(north.u, 0.0, 1.0), "{north:?}");
    assert!(close(north.n, 111.0, 3.0), "expected ~111 m north, got {}", north.n);

    // 0.001° east: positive East.
    let east = f.geodetic_to_enu(&Geodetic::from_degrees(40.0, -104.999, 1600.0));
    assert!(east.e > 0.0, "east should be +E, got {}", east.e);

    // +100 m altitude: ~+100 m Up.
    let up = f.geodetic_to_enu(&Geodetic { alt_hae_m: o.alt_hae_m + 100.0, ..o });
    assert!(close(up.u, 100.0, 1e-3), "u={}", up.u);
}

#[test]
fn enu_roundtrip() {
    let o = Geodetic::from_degrees(-33.8688, 151.2093, 58.0);
    let f = LocalFrame::new(o);
    let p = Enu { e: 1234.5, n: -6789.0, u: 321.0 };
    let p2 = f.geodetic_to_enu(&f.enu_to_geodetic(&p));
    assert!(
        close(p.e, p2.e, 1e-4) && close(p.n, p2.n, 1e-4) && close(p.u, p2.u, 1e-4),
        "{p:?} vs {p2:?}"
    );
}

#[test]
fn ned_at_the_origin_is_an_axis_swap() {
    let o = Geodetic::from_degrees(40.0, -105.0, 1600.0);
    let f = LocalFrame::new(o);
    let v = f.ned_to_enu(&o, &Ned { n: 100.0, e: 20.0, d: -3.0 });
    assert!(close(v.e, 20.0, 1e-9) && close(v.n, 100.0, 1e-9) && close(v.u, 3.0, 1e-9), "{v:?}");
}

#[test]
fn local_level_tilts_away_from_the_origin() {
    // 50 km east of the origin, "level" is tilted ~0.45° in the grid: a purely
    // horizontal 250 m/s eastbound velocity picks up a downward grid component.
    let o = Geodetic::from_degrees(40.0, -105.0, 0.0);
    let f = LocalFrame::new(o);
    let at = f.enu_to_geodetic(&Enu { e: 50_000.0, n: 0.0, u: 0.0 });
    let v = f.ned_to_enu(&at, &Ned { n: 0.0, e: 250.0, d: 0.0 });
    let tilt = (-v.u / 250.0).asin().to_degrees();
    assert!(close(tilt, 50.0 / 6371.0 * 57.2958, 0.02), "tilt {tilt}° ({v:?})");
    // Rotation preserves length and inverts exactly.
    assert!(close((v.e * v.e + v.n * v.n + v.u * v.u).sqrt(), 250.0, 1e-9));
    let back = f.enu_to_ned(&at, &v);
    assert!(close(back.e, 250.0, 1e-9) && close(back.n, 0.0, 1e-9) && close(back.d, 0.0, 1e-9));
}
