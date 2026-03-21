fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if target_os == "linux" {
        let compute = std::env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "87".to_string());
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={}", compute);

        if let Ok(cuda_path) = std::env::var("CUDA_HOME") {
            println!("cargo:rustc-env=CUDA_PATH={}", cuda_path);
        }
    }
}
