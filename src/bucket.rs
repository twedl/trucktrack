//! Stable truck → bucket assignment.
//!
//! Every stage writes `bucket=K/` folders keyed by this hash, and lookups
//! recompute it from a truck id, so it must never change: a different hash
//! moves trucks to other buckets and lookups miss data already written.
//! Polars' own `hash` is only stable within one Polars version, hence this.
//! The known values it must keep are pinned in `tests/test_buckets.py`.

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PySeries;

use crate::transform::polars_err;

/// Recorded in every `_buckets.json`; bump it, with a migration, if
/// `bucket_of` ever changes, so data laid out by the old hash is refused.
pub const HASH_VERSION: u32 = 1;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Bucket in `0..n_buckets` for a truck id.
///
/// FNV-1a over the id's UTF-8 bytes, then the MurmurHash3 finaliser so the
/// low bits the modulo keeps are well mixed.
#[must_use]
pub fn bucket_of(id: &str, n_buckets: u32) -> u32 {
    let mut h = FNV_OFFSET;
    for b in id.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    (h % u64::from(n_buckets)) as u32
}

fn check_n_buckets(n_buckets: u32) -> PyResult<()> {
    if n_buckets == 0 {
        return Err(PyValueError::new_err("n_buckets must be positive"));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "bucket_of")]
pub fn py_bucket_of(id: &str, n_buckets: u32) -> PyResult<u32> {
    check_n_buckets(n_buckets)?;
    Ok(bucket_of(id, n_buckets))
}

/// Bucket of each id in a string Series; nulls stay null.
///
/// Hashes without the GIL: polars calls this from several streaming
/// threads at once.
#[pyfunction]
pub fn bucket_ids(py: Python<'_>, ids: PySeries, n_buckets: u32) -> PyResult<PySeries> {
    check_n_buckets(n_buckets)?;
    let ids: Series = ids.into();
    let ca = ids.str().map_err(polars_err)?;
    let out: UInt32Chunked = py.detach(|| {
        ca.iter()
            .map(|id| id.map(|id| bucket_of(id, n_buckets)))
            .collect()
    });
    Ok(PySeries(out.with_name(ids.name().clone()).into_series()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_range() {
        for i in 0..1000 {
            assert!(bucket_of(&format!("truck-{i}"), 7) < 7);
        }
    }

    #[test]
    fn single_bucket() {
        assert_eq!(bucket_of("anything", 1), 0);
    }

    #[test]
    fn sequential_ids_spread_evenly() {
        let n = 16;
        let mut counts = vec![0u32; n as usize];
        for i in 0..16_000 {
            counts[bucket_of(&format!("truck-{i:06}"), n) as usize] += 1;
        }
        // 1000 expected per bucket; allow ±15%.
        assert!(
            counts.iter().all(|&c| (850..=1150).contains(&c)),
            "{counts:?}"
        );
    }

    #[test]
    fn zero_buckets_is_an_error() {
        assert!(py_bucket_of("a", 0).is_err());
    }
}
