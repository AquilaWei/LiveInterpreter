//! Which compute backend the accurate lane runs on.
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
//! * **Present and *real* are a third question**, and this one is currently
//!   answered upstream. Mesa serves Vulkan in software when no driver is
//!   reachable, and `llvmpipe` enumerates and runs; taking it would mean
//!   whisper.cpp on the CPU with "Vulkan" in every log line.
//!   [`is_software_rasterizer`] refuses it — but as of `whisper-rs-sys`
//!   0.15's ggml, so does ggml-vulkan itself ("If only CPU devices are
//!   available, return without devices"), and that is what was **measured**
//!   in a flatpak without `--device=dri`: no device, clean CPU fallback. The
//!   guard is here because that is an upstream implementation detail rather
//!   than a contract, and because the raw Vulkan enumeration in the very same
//!   sandbox *does* offer `llvmpipe`. It has not been seen to
//!   fire.
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
    /// 0.16, and OpenVINO only accelerates the encoder anyway.
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
    /// GPU this is system RAM, so it is not headroom the CPU also gets to use.
    pub total_bytes: usize,
}

/// What probing a backend told us.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Probe {
    Devices(Vec<GpuInfo>),
    /// This backend cannot be enumerated from here. Not the same as "no
    /// devices": the engine may still find one.
    Unavailable,
    /// The backend enumerated devices, and every one of them was a software
    /// rasteriser. Distinct from `Devices(vec![])` because the two want
    /// different things said: "no GPU here" is a fact about the machine, this
    /// is a driver or sandbox misconfiguration that would otherwise look like
    /// success. Carries the names, for the note.
    SoftwareOnly(Vec<String>),
}

/// Is this ggml device name a CPU pretending to be a GPU?
///
/// Mesa's `llvmpipe` (and `lavapipe`, its Vulkan face) implements the whole
/// API in software. Running the accurate lane on it means running whisper.cpp
/// on the CPU while every log line says "Vulkan" -- not slow, not broken, just
/// quietly answering a different question.
///
/// **Belt to ggml's braces, and say so.** A device that reaches this function
/// has already passed the type filter in [`ggml::devices`], and that filter
/// would not catch `llvmpipe`: ggml-vulkan reports every Vulkan device as
/// `GPU` or `IGPU`, never `CPU`, because `is_integrated_gpu` is the only
/// distinction it draws. What does catch it is one layer further out --
/// ggml-vulkan's own enumeration skips CPU-type physical devices entirely
/// ("If only CPU devices are available, return without devices",
/// `ggml-vulkan.cpp`), so the registry never offers one.
///
/// Measured in a flatpak without `--device=dri`: the accurate
/// lane fell back to the CPU with no Vulkan device at all, so **this has never
/// been seen to fire**. It stays because that upstream skip is an
/// implementation detail rather than a promise, and because the raw Vulkan
/// enumeration in that same sandbox does hand out `llvmpipe` -- the ICDs are
/// there, only ggml's filter stands between them and the accurate lane.
pub fn is_software_rasterizer(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    [
        "llvmpipe",
        "lavapipe",
        "softpipe",
        "swiftshader",
        "software rasterizer",
    ]
    .iter()
    .any(|s| n.contains(s))
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
/// and the settings UI.
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
/// one that works everywhere, but a vendor backend is only ever compiled
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
        let (gpus, software) = ggml::devices(accel);
        if gpus.is_empty() && !software.is_empty() {
            return Probe::SoftwareOnly(software);
        }
        Probe::Devices(gpus)
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

    /// The usable devices of one backend, and the names of any software
    /// rasterisers that were dropped.
    ///
    /// **The index is whisper.cpp's, not ours.** `whisper_backend_init_gpu`
    /// walks `ggml_backend_dev_get(i)` in order, counts every device of type
    /// GPU or IGPU *across all backends*, and takes `gpu_device` as an index
    /// into that count. So the counter here has to increment for devices this
    /// function then throws away -- otherwise rejecting `llvmpipe` at
    /// position 0 would hand back index 0 for the real GPU at position 1 and
    /// whisper.cpp would load the rasteriser we just refused.
    pub(super) fn devices(accel: Accel) -> (Vec<GpuInfo>, Vec<String>) {
        let mut out = Vec::new();
        let mut software = Vec::new();
        unsafe {
            let mut index = 0;
            for i in 0..sys::ggml_backend_dev_count() {
                let dev = sys::ggml_backend_dev_get(i);
                if dev.is_null() {
                    continue;
                }
                let kind = sys::ggml_backend_dev_type(dev);
                // An integrated GPU counts: it is the machine this project was
                // developed on (Arc 140V), and it measured 2.3x there.
                if kind != sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
                    && kind != sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU
                {
                    continue;
                }
                // Consumed whether or not this device survives the filters
                // below; see the note above.
                let here = index;
                index += 1;
                let reg = sys::ggml_backend_dev_backend_reg(dev);
                if reg.is_null() || !matches(accel, &cstr(sys::ggml_backend_reg_name(reg))) {
                    continue;
                }
                let name = cstr(sys::ggml_backend_dev_description(dev));
                if super::is_software_rasterizer(&name) {
                    software.push(name);
                    continue;
                }
                let (mut free, mut total) = (0usize, 0usize);
                sys::ggml_backend_dev_memory(dev, &mut free, &mut total);
                out.push(GpuInfo {
                    index: here,
                    name,
                    total_bytes: total,
                });
            }
        }
        (out, software)
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
        //
        // A backend that found only a software rasteriser keeps its reason so
        // the final CPU answer can give it. Nothing was asked for here, so
        // normally there is no note -- but this one is not "no GPU on this
        // machine", it is "something is misconfigured", and that is worth
        // saying even unasked.
        let mut fake = None;
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
                Probe::SoftwareOnly(names) => {
                    fake.get_or_insert_with(|| software_note(accel, &names));
                    continue;
                }
                Probe::Devices(_) => continue,
            }
        }
        return cpu(fake);
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
        Probe::SoftwareOnly(names) => cpu(Some(software_note(want, &names))),
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

/// The line for the status bar when the only device on offer was a fake one.
///
/// It names the device, because the fix depends on which situation this is:
/// inside a flatpak it means the manifest is missing `--device=dri`, and on a
/// bare system it means the real driver is not installed.
///
/// Deduplicated: a flatpak sandbox enumerates each physical device twice
/// (measured in the Flatpak sandbox), so the raw list says `llvmpipe, llvmpipe`.
fn software_note(accel: Accel, names: &[String]) -> String {
    let mut seen: Vec<&str> = Vec::new();
    for n in names {
        if !seen.contains(&n.as_str()) {
            seen.push(n);
        }
    }
    format!(
        "{} found only a software rasteriser ({}) -- that is the CPU with extra steps, \
         not a GPU; using the CPU path instead",
        accel.as_str(),
        seen.join(", ")
    )
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

    fn software() -> Probe {
        // Two entries, because that is what a flatpak sandbox actually
        // enumerates: it scans the ICDs twice.
        Probe::SoftwareOnly(vec![
            "llvmpipe (LLVM 21.1.1, 256 bits)".into(),
            "llvmpipe (LLVM 21.1.1, 256 bits)".into(),
        ])
    }

    #[test]
    fn a_software_rasteriser_is_refused_even_though_it_is_a_working_device() {
        // The whole point: llvmpipe answers every question correctly and is
        // still the wrong answer. Asked for vulkan, given llvmpipe, the right
        // move is the measured CPU path -- not "Vulkan" in the status bar.
        let s = select(DeviceRequest::Vulkan, &[Accel::Vulkan], |_| software());
        assert_eq!(s.accel, Accel::Cpu);
        let note = s.note.expect("a refusal has to say why");
        assert!(note.contains("llvmpipe"), "{note}");
        assert!(note.contains("software rasteriser"), "{note}");
    }

    #[test]
    fn the_note_names_the_device_once_however_many_times_it_was_enumerated() {
        let s = select(DeviceRequest::Vulkan, &[Accel::Vulkan], |_| software());
        let note = s.note.unwrap();
        assert_eq!(note.matches("llvmpipe").count(), 1, "{note}");
    }

    #[test]
    fn auto_says_why_it_fell_back_when_the_only_device_was_a_fake_one() {
        // `auto` asked for nothing, so it normally explains nothing. This case
        // is the exception: a machine offering nothing but a software
        // rasteriser is misconfigured somewhere, and the user cannot guess
        // that from silence. (Reaching it needs a ggml that hands the device
        // over; today's does not -- see `is_software_rasterizer`.)
        let s = select(DeviceRequest::Auto, &[Accel::Vulkan], |_| software());
        assert_eq!(s.accel, Accel::Cpu);
        assert!(s.note.unwrap().contains("llvmpipe"));
    }

    #[test]
    fn a_real_device_alongside_a_fake_one_is_still_used() {
        // `probe` only reports SoftwareOnly when nothing real survived, so a
        // machine with both keeps its GPU.
        let s = select(DeviceRequest::Vulkan, &[Accel::Vulkan], |_| {
            one("Intel(R) Graphics (LNL)")
        });
        assert_eq!(s.accel, Accel::Vulkan);
    }

    #[test]
    fn the_software_names_are_recognised_and_real_ones_are_not() {
        for name in [
            "llvmpipe (LLVM 21.1.1, 256 bits)",
            "lavapipe",
            "SwiftShader Device (Subzero)",
            "Software Rasterizer",
        ] {
            assert!(is_software_rasterizer(name), "{name}");
        }
        for name in [
            "Intel(R) Graphics (LNL)",
            "NVIDIA GeForce RTX 4090",
            "AMD Radeon RX 7900 XTX",
            "Apple M3 Pro",
        ] {
            assert!(!is_software_rasterizer(name), "{name}");
        }
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
