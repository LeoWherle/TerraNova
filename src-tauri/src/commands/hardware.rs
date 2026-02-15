use serde::Serialize;
use sysinfo::System;

#[derive(Serialize)]
pub struct HardwareInfo {
    pub cpu_cores: usize,
    pub cpu_name: String,
    pub total_ram_mb: u64,
}

#[tauri::command]
pub fn get_hardware_info() -> HardwareInfo {
    let sys = System::new_all();

    let cpu_cores = sys.cpus().len();
    let cpu_name = sys
        .cpus()
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_default();
    let total_ram_mb = sys.total_memory() / (1024 * 1024);

    HardwareInfo {
        cpu_cores,
        cpu_name,
        total_ram_mb,
    }
}
