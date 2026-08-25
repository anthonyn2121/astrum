//! Coordinate-conversion integration tests: WGS84 geodetic ↔ ECEF ↔ local-tangent
//! ENU round-trips and sign/scale checks, exercised through the crate's public API.

use astrum_geo::{ecef_to_geodetic, geodetic_to_ecef, Enu, Geodetic, LocalFrame};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn ecef_roundtrip() {
    let g = Geodetic::from_degrees(37.6188, -122.3756, 152.0);
    let g2 = ecef_to_geodetic(&geodetic_to_ecef(&g));
    assert!(close(g.lat_deg(), g2.lat_deg(), 1e-9), "{} vs {}", g.lat_deg(), g2.lat_deg());
    assert!(close(g.lon_deg(), g2.lon_deg(), 1e-9));
    assert!(close(g.alt_m, g2.alt_m, 1e-4), "{} vs {}", g.alt_m, g2.alt_m);
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
    let up = f.geodetic_to_enu(&Geodetic { alt_m: o.alt_m + 100.0, ..o });
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
