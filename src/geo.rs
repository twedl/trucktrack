/// Haversine distance between two (lat, lon) points in meters.
#[must_use]
pub fn haversine_distance_meters(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const R: f64 = 6_371_000.0; // Earth radius in meters

    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let lat1_r = lat1.to_radians();
    let lat2_r = lat2.to_radians();

    let a = (dlat / 2.0).sin().powi(2) + lat1_r.cos() * lat2_r.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * R * a.sqrt().asin()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_point_is_zero() {
        assert_eq!(haversine_distance_meters(40.0, -74.0, 40.0, -74.0), 0.0);
    }

    #[test]
    fn one_degree_latitude_at_equator() {
        let d = haversine_distance_meters(0.0, 0.0, 1.0, 0.0);
        // ~111.19 km
        assert!((d - 111_195.0).abs() < 100.0, "got {d}");
    }

    #[test]
    fn one_degree_longitude_at_equator() {
        let d = haversine_distance_meters(0.0, 0.0, 0.0, 1.0);
        assert!((d - 111_195.0).abs() < 100.0, "got {d}");
    }

    #[test]
    fn new_york_to_london() {
        // JFK (40.6413, -73.7781) to LHR (51.4700, -0.4543) ≈ 5,555 km
        let d = haversine_distance_meters(40.6413, -73.7781, 51.4700, -0.4543);
        assert!((d - 5_555_000.0).abs() < 20_000.0, "got {d}");
    }

    #[test]
    fn symmetry() {
        let d1 = haversine_distance_meters(37.7749, -122.4194, 34.0522, -118.2437);
        let d2 = haversine_distance_meters(34.0522, -118.2437, 37.7749, -122.4194);
        assert!((d1 - d2).abs() < 1e-6);
    }
}
