use pyo3::prelude::*;

use crate::geo::haversine_distance_meters;

// ── Constants ──────────────────────────────────────────────────────────

const LOCAL_KM: f64 = 100.0;
const REGIONAL_KM: f64 = 800.0;
const LONGHAUL_DEG: f64 = 8.0;
const VALHALLA_L1_DEG: f64 = 1.0;
const VALHALLA_L0_DEG: f64 = 4.0;
const TIER_NAMES: [&str; 3] = ["local", "regional", "longhaul"];

const HILBERT_ORDER: u8 = 12;
const LAT_MIN: f64 = 24.0;
const LAT_MAX: f64 = 84.0;
const LON_MIN: f64 = -141.0;
const LON_MAX: f64 = -52.0;

// ── Pure Rust helpers ──────────────────────────────────────────────────

fn valhalla_tile_id_inner(lat: f64, lon: f64, tile_deg: f64) -> u64 {
    let n_cols = (360.0 / tile_deg) as u64;
    let n_rows = (180.0 / tile_deg) as u64;
    let col = (((lon + 180.0) / tile_deg) as u64).min(n_cols - 1);
    let row = (((lat + 90.0) / tile_deg) as u64).min(n_rows - 1);
    row * n_cols + col
}

// ── PyO3 functions ─────────────────────────────────────────────────────

#[pyfunction]
pub fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    haversine_distance_meters(lat1, lon1, lat2, lon2) / 1000.0
}

#[pyfunction]
pub fn valhalla_tile_id(lat: f64, lon: f64, tile_deg: f64) -> u64 {
    valhalla_tile_id_inner(lat, lon, tile_deg)
}

#[pyfunction]
pub fn classify_and_partition_key(
    centroid_lat: f64,
    centroid_lon: f64,
    bbox_diag_km: f64,
) -> (&'static str, u64) {
    let (tier, tile_deg) = if bbox_diag_km < LOCAL_KM {
        (0u64, VALHALLA_L1_DEG)
    } else if bbox_diag_km < REGIONAL_KM {
        (1u64, VALHALLA_L0_DEG)
    } else {
        (2u64, LONGHAUL_DEG)
    };
    let tile = valhalla_tile_id_inner(centroid_lat, centroid_lon, tile_deg);
    let partition_id = (tier << 60) | tile;
    (TIER_NAMES[tier as usize], partition_id)
}

#[pyfunction]
pub fn hilbert_indices(lats: Vec<f64>, lons: Vec<f64>) -> PyResult<Vec<u64>> {
    if lats.len() != lons.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "lats and lons must have the same length",
        ));
    }
    let n = ((1u64 << HILBERT_ORDER) - 1) as f64;
    let lat_range = LAT_MAX - LAT_MIN;
    let lon_range = LON_MAX - LON_MIN;

    let result: Vec<u64> = lats
        .iter()
        .zip(lons.iter())
        .map(|(&la, &lo)| {
            let x = ((lo - LON_MIN) / lon_range * n).clamp(0.0, n) as u32;
            let y = ((la - LAT_MIN) / lat_range * n).clamp(0.0, n) as u32;
            fast_hilbert::xy2h(x, y, HILBERT_ORDER)
        })
        .collect();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haversine_same_point_is_zero() {
        assert_eq!(haversine_km(40.0, -74.0, 40.0, -74.0), 0.0);
    }

    #[test]
    fn haversine_known_distance() {
        // ~111.19 km for 1 degree latitude at equator
        let d = haversine_km(0.0, 0.0, 1.0, 0.0);
        assert!((d - 111.195).abs() < 0.1, "got {d}");
    }

    #[test]
    fn tile_id_origin() {
        // (-90, -180) should be tile 0 for any grid
        assert_eq!(valhalla_tile_id_inner(-90.0, -180.0, 1.0), 0);
        assert_eq!(valhalla_tile_id_inner(-90.0, -180.0, 4.0), 0);
    }

    #[test]
    fn tile_id_equator_prime_meridian() {
        // lat=0 → row = (0+90)/1 = 90, lon=0 → col = (0+180)/1 = 180
        // n_cols = 360, so tile = 90*360 + 180 = 32580
        assert_eq!(valhalla_tile_id_inner(0.0, 0.0, 1.0), 32580);
    }

    #[test]
    fn classify_local() {
        let (tier, pid) = classify_and_partition_key(43.65, -79.38, 10.0);
        assert_eq!(tier, "local");
        assert_eq!(pid >> 60, 0);
    }

    #[test]
    fn classify_regional() {
        let (tier, pid) = classify_and_partition_key(43.65, -79.38, 300.0);
        assert_eq!(tier, "regional");
        assert_eq!(pid >> 60, 1);
    }

    #[test]
    fn classify_longhaul() {
        let (tier, pid) = classify_and_partition_key(40.0, -100.0, 2000.0);
        assert_eq!(tier, "longhaul");
        assert_eq!(pid >> 60, 2);
    }

    #[test]
    fn classify_boundary_local_to_regional() {
        let (tier, _) = classify_and_partition_key(43.0, -79.0, LOCAL_KM);
        assert_eq!(tier, "regional");
    }

    #[test]
    fn classify_boundary_regional_to_longhaul() {
        let (tier, _) = classify_and_partition_key(43.0, -79.0, REGIONAL_KM);
        assert_eq!(tier, "longhaul");
    }

    #[test]
    fn hilbert_indices_basic() {
        let lats = vec![43.65, 40.0];
        let lons = vec![-79.38, -100.0];
        let result = hilbert_indices(lats, lons).unwrap();
        assert_eq!(result.len(), 2);
        // Values should be non-negative and fit in u64
        assert!(result[0] > 0);
        assert!(result[1] > 0);
        // Different coordinates should (usually) produce different indices
        assert_ne!(result[0], result[1]);
    }

    #[test]
    fn hilbert_indices_empty() {
        let result = hilbert_indices(vec![], vec![]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn hilbert_indices_length_mismatch() {
        let result = hilbert_indices(vec![1.0], vec![]);
        assert!(result.is_err());
    }
}
