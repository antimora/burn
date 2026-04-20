//! Shared TestBackend setup for benches.
//!
//! Mirrors `tests/common/backend.rs`: `TestBackend = burn_dispatch::Dispatch`, with a `#[ctor]`
//! that pins the default float/int dtypes to `f32`/`i32` so backends that advertise a different
//! default (e.g. bf16) don't silently convert bench inputs.

use burn_tensor::backend::Backend;
use ctor::ctor;

pub type FloatElem = f32;
pub type IntElem = i32;
pub type TestBackend = burn_dispatch::Dispatch;

#[ctor]
fn init_device_settings() {
    let device = burn_dispatch::DispatchDevice::default();
    burn_tensor::set_default_dtypes::<TestBackend>(
        &device,
        <FloatElem as burn_tensor::Element>::dtype(),
        <IntElem as burn_tensor::Element>::dtype(),
    )
    .unwrap();
}

/// Block until all outstanding ops on the default device complete.
///
/// GPU backends (cuda, wgpu, rocm, metal, vulkan) dispatch ops asynchronously; without a sync
/// barrier inside the timed region a bench would measure dispatch latency, not execution time.
/// On CPU backends (flex, ndarray) this is a no-op via the `Backend::sync` default.
#[inline]
pub fn sync() {
    TestBackend::sync(&Default::default()).unwrap();
}

/// Extension trait adding a synced variant of `Bencher::bench`.
///
/// `bench_synced` runs the op then forces a device sync before returning, so the timed region
/// covers actual execution on async backends.
pub trait BencherExt<'a, 'b> {
    fn bench_synced<O, F>(self, benched: F)
    where
        F: Fn() -> O + Sync;
}

impl<'a, 'b> BencherExt<'a, 'b> for divan::Bencher<'a, 'b> {
    #[inline]
    fn bench_synced<O, F>(self, benched: F)
    where
        F: Fn() -> O + Sync,
    {
        self.bench(move || {
            let r = benched();
            sync();
            r
        });
    }
}
