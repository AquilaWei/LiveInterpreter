//! Which compute backend the accurate lane runs on (PLAN §8.1).
//!
//! The rule the plan sets is "probe at startup, pick a backend, fall back to
//! CPU on failure, and report what was actually chosen". Two details make that
//! less obvious than it sounds:
//!
//! * **Compiled-in and present are different questions.** A binary built with
//!   `gpu-vulkan` still has to run on machines with no Vulkan device, and a
//!   binary built without it must not pretend. So the policy takes both the
//!   feature set and the probe result, and [`select`] is a pure function over
//!   them — the part that can be tested on a machine with no GPU at all.
//! * **The enumeration comes from ggml itself**, through its backend-device
//!   registry — the same registry whisper.cpp picks the device from, so it
//!   cannot disagree with the engine about what exists. That covers every
//!   backend at once, Vulkan and the vendor ones alike; guessing from the
//!   presence of a driver library would have been a guess.
//!
//!   (`whisper_rs::vulkan::list_devices` looks like the intended way to do
//!   this and is not usable: as of whisper-rs 0.16 it still imports
//!   `ggml_backend_vk_*`, which ggml replaced with the generic registry, so the
//!   crate's `vulkan` feature does not compile at all. `li-asr` therefore turns
//!   the GPU backends on through `whisper-rs-sys` and reads the registry here.)
//!
//! [`Probe::Unavailable`] is left for the case that remains: a build with no
//! whisper.cpp in it, which has no registry to ask.

use serde::{Deserialize, Serialize};

/// A compute backend for the accurate lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Accel {
    Cpu,
    Vulkan,
    Cuda,
    /// Intel oneAPI SYCL. whisper.cpp has no OpenVINO path in `whisper-rs`
    /// 0.16, and OpenVINO only accelerates the encoder anyway (§8.1).
    Sycl,
    /// AMD ROCm, via hipBLAS.
    Hip,
}

impl Accel {
    pub fn as_str(self) -> &'static str {
        match self {
            Accel::Cpu => "cpu",
            Accel::Vulkan => "vulkan",
            Accel::Cuda => "cuda",
            Accel::Sycl => "sycl",
            Accel::Hip => "hip",
        }
    }
}

/// `[asr] device` in `config.toml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceRequest {
    #[default]
    Auto,
    Cpu,
    Vulkan,
    Cuda,
    Sycl,
    Hip,
}

impl DeviceRequest {
    fn as_accel(self) -> Option<Accel> {
        match self {
            DeviceRequest::Auto => None,
            DeviceRequest::Cpu => Some(Accel::Cpu),
            DeviceRequest::Vulkan => Some(Accel::Vulkan),
            DeviceRequest::Cuda => Some(Accel::Cuda),
            DeviceRequest::Sycl => Some(Accel::Sycl),
            DeviceRequest::Hip => Some(Accel::Hip),
        }
    }
}

impl std::str::FromStr for DeviceRequest {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "auto" => DeviceRequest::Auto,
            "cpu" => DeviceRequest::Cpu,
            "vulkan" => DeviceRequest::Vulkan,
            "cuda" => DeviceRequest::Cuda,
            "sycl" => DeviceRequest::Sycl,
            "hip" | "rocm" | "hipblas" => DeviceRequest::Hip,
            other => anyhow::bail!(
                "unknown asr device {other:?} \
                 (want auto | cpu | vulkan | cuda | sycl | hip)"
            ),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index: i32,
    pub name: String,
    /// Total device memory in bytes, as ggml reports it. On a UMA integrated
    /// GPU this is system RAM, so it is not headroom the CPU also gets to use
    /// (task 1.0b).
    pub total_bytes: usize,
}

/// What probing a backend told us.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Probe {
    Devices(Vec<GpuInfo>),
    /// This backend cannot be enumerated from here. Not the same as "no
    /// devices": the engine may still find one.
    Unavailable,
}

/// The chosen backend, and what to say about it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Selection {
    pub accel: Accel,
    /// Index into the backend's own device list; meaningless for `Cpu`.
    pub gpu_index: i32,
    pub device: String,
    /// Set when the answer is not what was asked for. This is the line worth
    /// putting in the status bar.
    pub note: Option<String>,
}

/// What this engine loaded and what it is running on. Reaches `EngineStatus`
/// and the settings UI (PLAN §8.1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendInfo {
    pub engine: &'static str,
    pub model: String,
    pub selection: Selection,
}

impl std::fmt::Display for BackendInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} on {} ({})",
            self.engine,
            self.model,
            self.selection.device,
            self.selection.accel.as_str()
        )?;
        if let Some(note) = &self.selection.note {
            write!(f, " -- {note}")?;
        }
        Ok(())
    }
}

/// Backends this binary was built with, most preferred first.
///
/// Vulkan comes last on purpose. It is the *default* backend because it is the
/// one that works everywhere (§8.1), but a vendor backend is only ever compiled
/// in because somebody asked for it at build time, and ignoring that under
/// `auto` would make the feature flag do nothing.
pub fn compiled_accels() -> Vec<Accel> {
    let mut v = Vec::new();
    if cfg!(feature = "gpu-cuda") {
        v.push(Accel::Cuda);
    }
    if cfg!(feature = "gpu-hipblas") {
        v.push(Accel::Hip);
    }
    if cfg!(feature = "gpu-sycl") {
        v.push(Accel::Sycl);
    }
    if cfg!(feature = "gpu-vulkan") {
        v.push(Accel::Vulkan);
    }
    v
}

/// Ask the engine itself what devices it can see.
pub fn probe(accel: Accel) -> Probe {
    if accel == Accel::Cpu {
        return Probe::Devices(Vec::new());
    }
    #[cfg(feature = "whispercpp")]
    {
        Probe::Devices(ggml::devices(accel))
    }
    #[cfg(not(feature = "whispercpp"))]
    {
        Probe::Unavailable
    }
}

/// ggml's backend-device registry.
///
/// A backend registers itself only if whisper.cpp was built with it, so a
/// device appearing here means the engine can actually use it. Devices are
/// numbered per backend, because that is the numbering `use_gpu`/`gpu_device`
/// expects.
#[cfg(feature = "whispercpp")]
mod ggml {
    // Enumerating C globals. Every pointer here comes from the registry, is
    // read immediately, and is never stored.
    #![allow(unsafe_code)]

    use std::ffi::CStr;

    use whisper_rs_sys as sys;

    use super::{Accel, GpuInfo};

    /// What ggml calls each backend in its registry.
    fn matches(accel: Accel, reg_name: &str) -> bool {
        let name = reg_name.to_ascii_lowercase();
        match accel {
            Accel::Vulkan => name == "vulkan",
            Accel::Cuda => name == "cuda",
            Accel::Sycl => name == "sycl",
            // AMD's backend has gone by both names across ggml releases.
            Accel::Hip => name == "rocm" || name == "hip",
            Accel::Cpu => false,
        }
    }

    pub(super) fn devices(accel: Accel) -> Vec<GpuInfo> {
        let mut out = Vec::new();
        unsafe {
            let mut index = 0;
            for i in 0..sys::ggml_backend_dev_count() {
                let dev = sys::ggml_backend_dev_get(i);
                if dev.is_null() {
                    continue;
                }
                let kind = sys::ggml_backend_dev_type(dev);
                // An integrated GPU counts: it is the machine this project was
                // developed on (Arc 140V), and task 1.0b measured 2.3x on it.
                if kind != sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
                    && kind != sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU
                {
                    continue;
                }
                let reg = sys::ggml_backend_dev_backend_reg(dev);
                if reg.is_null() || !matches(accel, &cstr(sys::ggml_backend_reg_name(reg))) {
                    continue;
                }
                let (mut free, mut total) = (0usize, 0usize);
                sys::ggml_backend_dev_memory(dev, &mut free, &mut total);
                out.push(GpuInfo {
                    index,
                    name: cstr(sys::ggml_backend_dev_description(dev)),
                    total_bytes: total,
                });
                index += 1;
            }
        }
        out
    }

    unsafe fn cstr(p: *const std::os::raw::c_char) -> String {
        if p.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() }
    }
}

/// Resolve a request against what is compiled in and what is present.
///
/// Never an error: a config file that names a backend this machine cannot
/// provide should start the program on the CPU and say so, not refuse to start.
pub fn select(req: DeviceRequest, compiled: &[Accel], probe: impl Fn(Accel) -> Probe) -> Selection {
    let cpu = |note: Option<String>| Selection {
        accel: Accel::Cpu,
        gpu_index: 0,
        device: "CPU".into(),
        note,
    };

    let Some(want) = req.as_accel() else {
        // auto: first compiled backend with a device, else CPU.
        for &accel in compiled {
            match probe(accel) {
                Probe::Devices(gpus) if !gpus.is_empty() => return gpu(accel, &gpus[0], None),
                // Not enumerable, but compiled in deliberately: let the engine try.
                Probe::Unavailable => {
                    return Selection {
                        accel,
                        gpu_index: 0,
                        device: format!("{} device 0", accel.as_str()),
                        note: Some(format!(
                            "{} cannot be enumerated from here; the engine picks the device",
                            accel.as_str()
                        )),
                    };
                }
                Probe::Devices(_) => continue,
            }
        }
        return cpu(None);
    };

    if want == Accel::Cpu {
        return cpu(None);
    }
    if !compiled.contains(&want) {
        return cpu(Some(format!(
            "asked for {}, but this build has no {} backend (feature `gpu-{}`); using the CPU",
            want.as_str(),
            want.as_str(),
            want.as_str()
        )));
    }
    match probe(want) {
        Probe::Devices(gpus) if !gpus.is_empty() => gpu(want, &gpus[0], None),
        Probe::Devices(_) => cpu(Some(format!(
            "asked for {}, but no {} device is present; using the CPU",
            want.as_str(),
            want.as_str()
        ))),
        Probe::Unavailable => Selection {
            accel: want,
            gpu_index: 0,
            device: format!("{} device 0", want.as_str()),
            note: Some(format!(
                "{} cannot be enumerated from here; the engine picks the device",
                want.as_str()
            )),
        },
    }
}

fn gpu(accel: Accel, info: &GpuInfo, note: Option<String>) -> Selection {
    Selection {
        accel,
        gpu_index: info.index,
        device: info.name.clone(),
        note,
    }
}

/// The whole startup decision, ready for [`BackendInfo`].
pub fn resolve(req: DeviceRequest) -> Selection {
    select(req, &compiled_accels(), probe)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(name: &str) -> Probe {
        Probe::Devices(vec![GpuInfo {
            index: 0,
            name: name.into(),
            total_bytes: 8 << 30,
        }])
    }
    fn none() -> Probe {
        Probe::Devices(Vec::new())
    }

    #[test]
    fn auto_takes_the_first_compiled_backend_that_has_a_device() {
        let s = select(DeviceRequest::Auto, &[Accel::Vulkan], |_| one("Arc 140V"));
        assert_eq!(s.accel, Accel::Vulkan);
        assert_eq!(s.device, "Arc 140V");
        assert_eq!(s.note, None);
    }

    #[test]
    fn auto_falls_through_a_compiled_backend_with_no_device() {
        let s = select(DeviceRequest::Auto, &[Accel::Vulkan], |_| none());
        assert_eq!(s.accel, Accel::Cpu);
        // Nothing was asked for, so nothing was denied: no note.
        assert_eq!(s.note, None);
    }

    #[test]
    fn auto_on_a_cpu_only_build_is_the_cpu() {
        let s = select(DeviceRequest::Auto, &[], |_| one("should not be asked"));
        assert_eq!(s.accel, Accel::Cpu);
    }

    #[test]
    fn a_backend_compiled_out_falls_back_and_says_which_feature_is_missing() {
        let s = select(DeviceRequest::Vulkan, &[], |_| one("Arc 140V"));
        assert_eq!(s.accel, Accel::Cpu);
        assert!(s.note.unwrap().contains("gpu-vulkan"));
    }

    #[test]
    fn a_backend_with_no_device_falls_back_and_says_so() {
        let s = select(DeviceRequest::Vulkan, &[Accel::Vulkan], |_| none());
        assert_eq!(s.accel, Accel::Cpu);
        assert!(s.note.unwrap().contains("no vulkan device"));
    }

    #[test]
    fn an_unenumerable_backend_is_used_rather_than_declared_absent() {
        // Reachable when whisper.cpp is compiled out, so there is no registry
        // to ask. Refusing the backend would be claiming knowledge we do not
        // have; letting the engine try and saying so is the honest answer.
        let s = select(DeviceRequest::Cuda, &[Accel::Cuda], |_| Probe::Unavailable);
        assert_eq!(s.accel, Accel::Cuda);
        assert!(s.note.is_some());
    }

    #[test]
    fn the_real_probe_answers_for_every_backend_without_panicking() {
        // ggml's registry is global state initialised on first use; asking it
        // before any model is loaded must be safe.
        for a in [
            Accel::Cpu,
            Accel::Vulkan,
            Accel::Cuda,
            Accel::Sycl,
            Accel::Hip,
        ] {
            let _ = probe(a);
        }
    }

    #[test]
    fn a_vendor_backend_outranks_vulkan_under_auto() {
        let compiled = [Accel::Cuda, Accel::Vulkan];
        let s = select(DeviceRequest::Auto, &compiled, |a| match a {
            Accel::Cuda => one("RTX 4090"),
            _ => one("llvmpipe"),
        });
        assert_eq!(s.accel, Accel::Cuda);
    }

    #[test]
    fn asking_for_the_cpu_is_honoured_even_with_a_gpu_present() {
        let s = select(DeviceRequest::Cpu, &[Accel::Vulkan], |_| one("Arc 140V"));
        assert_eq!(s.accel, Accel::Cpu);
        assert_eq!(s.note, None);
    }

    #[test]
    fn device_names_round_trip_through_the_config_file() {
        use std::str::FromStr;
        for (s, want) in [
            ("auto", DeviceRequest::Auto),
            ("cpu", DeviceRequest::Cpu),
            ("vulkan", DeviceRequest::Vulkan),
            ("rocm", DeviceRequest::Hip),
        ] {
            assert_eq!(DeviceRequest::from_str(s).unwrap(), want);
        }
        assert!(DeviceRequest::from_str("opencl").is_err());
    }
}
