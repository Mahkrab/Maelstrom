//! CPU builds never inspect the CUDA environment. The optional bridge uses a
//! system toolkit and embeds PTX, so execution needs no source or toolkit paths.

use std::{env, fs, path::PathBuf, process::Command};

fn run(command: &mut Command) {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("could not execute {command:?}: {error}; see docs/cuda.md"));
    assert!(
        output.status.success(),
        "{command:?} failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    assert_eq!(
        env::var("CARGO_CFG_TARGET_OS").as_deref(),
        Ok("linux"),
        "CUDA currently supports Linux/WSL only; see docs/cuda.md"
    );
    assert_eq!(
        env::var("CARGO_CFG_TARGET_ARCH").as_deref(),
        Ok("x86_64"),
        "CUDA currently supports x86_64 only; see docs/cuda.md"
    );
    assert_eq!(
        env::var("HOST"),
        env::var("TARGET"),
        "CUDA cross-compilation is unsupported; see docs/cuda.md"
    );
    for variable in ["CUDA_HOME", "MAELSTROM_CUDA_ARCH", "CXX", "AR"] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    for file in [
        "native/bridge.cpp",
        "native/probe.h",
        "native/roundtrip.cu",
        "native/solver.h",
        "native/solver.cu",
        "native/kernels.cu",
        "native/power.cuh",
    ] {
        println!("cargo:rerun-if-changed={file}");
    }
    let toolkit =
        PathBuf::from(env::var_os("CUDA_HOME").unwrap_or_else(|| "/usr/local/cuda".into()));
    assert!(
        toolkit.is_absolute(),
        "CUDA_HOME must be an absolute toolkit directory"
    );
    assert!(
        toolkit.join("bin/nvcc").is_file() && toolkit.join("include/cuda.h").is_file(),
        "CUDA feature requires nvcc and cuda.h in CUDA_HOME (default /usr/local/cuda); see docs/cuda.md"
    );
    // A virtual target permits a deliberate compatible-device policy at runtime.
    // Do not accept arbitrary compiler arguments from this single-value setting.
    let arch = env::var("MAELSTROM_CUDA_ARCH").unwrap_or_else(|_| "89".into());
    assert!(
        (2..=3).contains(&arch.len()) && arch.bytes().all(|b| b.is_ascii_digit()),
        "MAELSTROM_CUDA_ARCH must be a numeric compute capability, e.g. 89"
    );
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo OUT_DIR"));
    let ptx = out.join("roundtrip.ptx");
    run(Command::new(toolkit.join("bin/nvcc"))
        .args([
            "--ptx",
            "--std=c++17",
            "--fmad=false",
            "--ftz=false",
            "--prec-div=true",
            "--prec-sqrt=true",
            "-O3",
        ])
        .arg(format!("--gpu-architecture=compute_{arch}"))
        .arg("native/kernels.cu")
        .arg("-o")
        .arg(&ptx));
    let mut bytes = fs::read(&ptx).expect("compiled PTX");
    bytes.push(0);
    fs::write(&ptx, bytes).expect("NUL-terminated PTX");
    run(
        Command::new(env::var_os("CXX").unwrap_or_else(|| "c++".into()))
            .args([
                "-std=c++17",
                "-O2",
                "-fPIC",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-c",
                "native/bridge.cpp",
                "-o",
            ])
            .arg(out.join("bridge.o"))
            .arg("-I")
            .arg(toolkit.join("include")),
    );
    run(
        Command::new(env::var_os("AR").unwrap_or_else(|| "ar".into()))
            .arg("crs")
            .arg(out.join("libmaelstrom_cuda.a"))
            .arg(out.join("bridge.o")),
    );
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=maelstrom_cuda");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-env=MAELSTROM_CUDA_ARCH={arch}");
}
