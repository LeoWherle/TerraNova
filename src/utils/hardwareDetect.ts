import { invoke } from "@tauri-apps/api/core";

export interface HardwareInfo {
  cpuCores: number;
  cpuName: string;
  totalRamMb: number;
  gpuRenderer: string;
  gpuVendor: string;
  estimatedVramMb: number;
}

let _cached: HardwareInfo | null = null;

export async function detectHardware(): Promise<HardwareInfo> {
  if (_cached) return _cached;

  // System info from Tauri (CPU + RAM)
  let cpuCores = navigator.hardwareConcurrency || 4;
  let cpuName = "";
  let totalRamMb = 8192; // fallback

  try {
    const info = await invoke<{ cpu_cores: number; cpu_name: string; total_ram_mb: number }>("get_hardware_info");
    cpuCores = info.cpu_cores || cpuCores;
    cpuName = info.cpu_name || "";
    totalRamMb = info.total_ram_mb || totalRamMb;
  } catch {
    // Tauri command unavailable — use web fallback
  }

  // GPU info from WebGL
  let gpuRenderer = "";
  let gpuVendor = "";
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (gl) {
      const ext = gl.getExtension("WEBGL_debug_renderer_info");
      if (ext) {
        gpuRenderer = gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) || "";
        gpuVendor = gl.getParameter(ext.UNMASKED_VENDOR_WEBGL) || "";
      }
      // Clean up
      const loseCtx = gl.getExtension("WEBGL_lose_context");
      loseCtx?.loseContext();
    }
  } catch {
    // WebGL unavailable
  }

  const estimatedVramMb = estimateVram(gpuRenderer);

  _cached = { cpuCores, cpuName, totalRamMb, gpuRenderer, gpuVendor, estimatedVramMb };
  return _cached;
}

function estimateVram(renderer: string): number {
  if (!renderer) return 4096;
  const r = renderer.toLowerCase();

  // NVIDIA
  if (r.includes("rtx 40")) return 16384;
  if (r.includes("rtx 30")) return r.includes("3060") ? 12288 : r.includes("3070") ? 8192 : r.includes("3080") ? 10240 : r.includes("3090") ? 24576 : 8192;
  if (r.includes("rtx 20")) return 8192;
  if (r.includes("gtx 16")) return 6144;
  if (r.includes("gtx 10")) return r.includes("1080") ? 8192 : r.includes("1070") ? 8192 : r.includes("1060") ? 6144 : 4096;

  // AMD
  if (r.includes("rx 7")) return r.includes("7900") ? 24576 : 16384;
  if (r.includes("rx 6")) return r.includes("6900") ? 16384 : r.includes("6800") ? 16384 : r.includes("6700") ? 12288 : 8192;
  if (r.includes("rx 5")) return 8192;

  // Apple Silicon
  if (r.includes("apple m4")) return 32768;
  if (r.includes("apple m3")) return r.includes("max") ? 40960 : r.includes("pro") ? 18432 : 10240;
  if (r.includes("apple m2")) return r.includes("max") ? 32768 : r.includes("pro") ? 16384 : 8192;
  if (r.includes("apple m1")) return r.includes("max") ? 32768 : r.includes("pro") ? 16384 : 8192;
  if (r.includes("apple")) return 8192;

  // Intel integrated
  if (r.includes("intel") && (r.includes("iris") || r.includes("uhd") || r.includes("hd"))) return 2048;

  return 4096;
}
