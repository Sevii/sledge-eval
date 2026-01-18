"""Hardware information extraction from llama-server logs."""

import re
import platform
from pathlib import Path
from typing import Dict, Optional, Any
from pydantic import BaseModel


class HardwareInfo(BaseModel):
    """Hardware information extracted from llama-server logs and system."""
    
    # GPU Information (from llama-server logs)
    gpu_name: Optional[str] = None
    gpu_family: Optional[str] = None
    metal_backend: Optional[bool] = None
    has_unified_memory: Optional[bool] = None
    has_bfloat: Optional[bool] = None
    has_tensor: Optional[bool] = None
    recommended_max_working_set_size_mb: Optional[float] = None
    
    # Memory Information (from llama-server logs)
    total_memory_mb: Optional[float] = None
    free_memory_mb: Optional[float] = None
    model_memory_mb: Optional[float] = None
    context_memory_mb: Optional[float] = None
    compute_memory_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    
    # Threading Information (from llama-server logs)
    n_threads: Optional[int] = None
    n_threads_batch: Optional[int] = None
    total_threads: Optional[int] = None
    http_server_threads: Optional[int] = None
    
    # Model Information (from llama-server logs)
    model_size_gb: Optional[float] = None
    model_params_b: Optional[float] = None
    context_size: Optional[int] = None
    batch_size: Optional[int] = None
    
    # System Information (from platform)
    os_name: Optional[str] = None
    os_version: Optional[str] = None
    architecture: Optional[str] = None
    processor: Optional[str] = None
    
    # Build Information (from llama-server logs)
    llama_cpp_build: Optional[str] = None
    llama_cpp_commit: Optional[str] = None
    compiler: Optional[str] = None


class HardwareDetector:
    """Extracts hardware information from llama-server logs and system."""
    
    def __init__(self, log_file_path: Optional[Path] = None):
        """
        Initialize hardware detector.
        
        Args:
            log_file_path: Path to llama-server.log file. If None, uses default.
        """
        self.log_file_path = log_file_path or Path("llama-server.log")
    
    def extract_hardware_info(self) -> HardwareInfo:
        """Extract comprehensive hardware information."""
        info = HardwareInfo()
        
        # Get system information
        self._extract_system_info(info)
        
        # Extract from llama-server logs if available
        if self.log_file_path.exists():
            self._extract_log_info(info)
        
        return info
    
    def _extract_system_info(self, info: HardwareInfo) -> None:
        """Extract system information using platform module."""
        try:
            info.os_name = platform.system()
            info.os_version = platform.release()
            info.architecture = platform.machine()
            info.processor = platform.processor()

            # On Linux, get more detailed CPU info from /proc/cpuinfo
            if info.os_name == "Linux":
                self._extract_linux_cpu_info(info)
                self._extract_linux_memory_info(info)
                self._extract_nvidia_gpu_info(info)
        except (OSError, AttributeError) as e:
            # Log but continue if system info extraction fails
            import logging
            logging.getLogger(__name__).debug(f"System info extraction failed: {e}")

    def _extract_linux_cpu_info(self, info: HardwareInfo) -> None:
        """Extract CPU information from /proc/cpuinfo on Linux."""
        try:
            cpuinfo_path = Path("/proc/cpuinfo")
            if cpuinfo_path.exists():
                content = cpuinfo_path.read_text()
                # Get model name
                model_match = re.search(r'model name\s*:\s*(.+)', content)
                if model_match:
                    info.processor = model_match.group(1).strip()
        except (OSError, IOError) as e:
            import logging
            logging.getLogger(__name__).debug(f"CPU info extraction failed: {e}")

    def _extract_linux_memory_info(self, info: HardwareInfo) -> None:
        """Extract memory information from /proc/meminfo on Linux."""
        try:
            meminfo_path = Path("/proc/meminfo")
            if meminfo_path.exists():
                content = meminfo_path.read_text()
                # Get total memory (in kB, convert to MB)
                mem_match = re.search(r'MemTotal:\s*(\d+)\s*kB', content)
                if mem_match:
                    info.total_memory_mb = float(mem_match.group(1)) / 1024
                # Get available memory
                avail_match = re.search(r'MemAvailable:\s*(\d+)\s*kB', content)
                if avail_match:
                    info.free_memory_mb = float(avail_match.group(1)) / 1024
        except (OSError, IOError, ValueError) as e:
            import logging
            logging.getLogger(__name__).debug(f"Memory info extraction failed: {e}")

    def _extract_nvidia_gpu_info(self, info: HardwareInfo) -> None:
        """Extract NVIDIA GPU information using nvidia-smi."""
        import subprocess
        try:
            # Get GPU name and memory
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().split('\n')[0]  # First GPU
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 1 and not info.gpu_name:
                    info.gpu_name = parts[0]
                if len(parts) >= 2:
                    info.gpu_memory_mb = float(parts[1])
                if len(parts) >= 3:
                    info.free_memory_mb = float(parts[2])
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            import logging
            logging.getLogger(__name__).debug(f"NVIDIA GPU info extraction failed: {e}")
    
    def _extract_log_info(self, info: HardwareInfo) -> None:
        """Extract information from llama-server logs."""
        try:
            with open(self.log_file_path, 'r') as f:
                log_content = f.read()
            
            # Extract GPU information
            self._extract_gpu_info(log_content, info)
            
            # Extract memory information
            self._extract_memory_info(log_content, info)
            
            # Extract threading information
            self._extract_thread_info(log_content, info)
            
            # Extract model information
            self._extract_model_info(log_content, info)
            
            # Extract build information
            self._extract_build_info(log_content, info)

        except (OSError, IOError, ValueError) as e:
            import logging
            logging.getLogger(__name__).debug(f"Log file parsing failed: {e}")
    
    def _extract_gpu_info(self, log_content: str, info: HardwareInfo) -> None:
        """Extract GPU information from logs."""
        # Metal backend (macOS)
        # GPU name: "ggml_metal_device_init: GPU name:   Apple M3"
        gpu_name_match = re.search(r'ggml_metal_device_init: GPU name:\s+(.+)', log_content)
        if gpu_name_match:
            info.gpu_name = gpu_name_match.group(1).strip()

        # GPU family: "ggml_metal_device_init: GPU family: MTLGPUFamilyApple9  (1009)"
        gpu_family_match = re.search(r'ggml_metal_device_init: GPU family:\s+(\w+)', log_content)
        if gpu_family_match:
            info.gpu_family = gpu_family_match.group(1).strip()

        # Metal backend detection
        if 'ggml_metal_device_init' in log_content:
            info.metal_backend = True

        # Capabilities
        unified_memory_match = re.search(r'ggml_metal_device_init: has unified memory\s+=\s+(true|false)', log_content)
        if unified_memory_match:
            info.has_unified_memory = unified_memory_match.group(1) == 'true'

        bfloat_match = re.search(r'ggml_metal_device_init: has bfloat\s+=\s+(true|false)', log_content)
        if bfloat_match:
            info.has_bfloat = bfloat_match.group(1) == 'true'

        tensor_match = re.search(r'ggml_metal_device_init: has tensor\s+=\s+(true|false)', log_content)
        if tensor_match:
            info.has_tensor = tensor_match.group(1) == 'true'

        # Working set size: "ggml_metal_device_init: recommendedMaxWorkingSetSize  = 17179.89 MB"
        working_set_match = re.search(r'recommendedMaxWorkingSetSize\s+=\s+([\d.]+)\s+MB', log_content)
        if working_set_match:
            info.recommended_max_working_set_size_mb = float(working_set_match.group(1))
            # Use recommendedMaxWorkingSetSize as GPU memory available
            info.gpu_memory_mb = float(working_set_match.group(1))

        # Vulkan backend (Linux/Windows) - fallback if GPU name not already set
        # Format: "ggml_vulkan: 0 = NVIDIA GeForce RTX 5070 Ti (NVIDIA) | uma: 0 | ..."
        if not info.gpu_name:
            vulkan_match = re.search(r'ggml_vulkan: 0 = ([^|]+)\|', log_content)
            if vulkan_match:
                gpu_info = vulkan_match.group(1).strip()
                # Remove vendor suffix in parentheses if present
                gpu_info = re.sub(r'\s*\([^)]+\)\s*$', '', gpu_info)
                info.gpu_name = gpu_info.strip()

        # CUDA backend (Linux/Windows)
        # Format varies, try common patterns
        if not info.gpu_name:
            cuda_match = re.search(r'ggml_cuda_init: found (\d+) CUDA devices?:\s*\n\s*Device 0: (.+)', log_content)
            if cuda_match:
                info.gpu_name = cuda_match.group(2).strip()
    
    def _extract_memory_info(self, log_content: str, info: HardwareInfo) -> None:
        """Extract memory information from logs."""
        # Memory breakdown: "llama_memory_breakdown_print: |   - Metal (Apple M3)   | 16384 = 12803 + (2717 =  2039 +     416 +     262) +         862 |"
        memory_breakdown_match = re.search(
            r'llama_memory_breakdown_print:.*Metal.*\|\s+(\d+)\s+=\s+(\d+)\s+\+\s+\((\d+)\s+=\s+(\d+)\s+\+\s+(\d+)\s+\+\s+(\d+)\)',
            log_content
        )
        if memory_breakdown_match:
            info.total_memory_mb = float(memory_breakdown_match.group(1))
            info.free_memory_mb = float(memory_breakdown_match.group(2))
            # Group 3 is used memory, Group 4 is model, Group 5 is context, Group 6 is compute
            info.model_memory_mb = float(memory_breakdown_match.group(4))
            info.context_memory_mb = float(memory_breakdown_match.group(5))
            info.compute_memory_mb = float(memory_breakdown_match.group(6))
    
    def _extract_thread_info(self, log_content: str, info: HardwareInfo) -> None:
        """Extract threading information from logs."""
        # Threading: "system info: n_threads = 4, n_threads_batch = 4, total_threads = 8"
        thread_match = re.search(r'n_threads = (\d+), n_threads_batch = (\d+), total_threads = (\d+)', log_content)
        if thread_match:
            info.n_threads = int(thread_match.group(1))
            info.n_threads_batch = int(thread_match.group(2))
            info.total_threads = int(thread_match.group(3))
        
        # HTTP server threads: "init: using 7 threads for HTTP server"
        http_threads_match = re.search(r'init: using (\d+) threads for HTTP server', log_content)
        if http_threads_match:
            info.http_server_threads = int(http_threads_match.group(1))
    
    def _extract_model_info(self, log_content: str, info: HardwareInfo) -> None:
        """Extract model information from logs."""
        # Model size: "print_info: file size   = 1.99 GiB (4.99 BPW)"
        model_size_match = re.search(r'print_info: file size\s+=\s+([\d.]+)\s+GiB', log_content)
        if model_size_match:
            info.model_size_gb = float(model_size_match.group(1))
        
        # Model parameters: "print_info: model params     = 3.43 B"
        params_match = re.search(r'print_info: model params\s+=\s+([\d.]+)\s+B', log_content)
        if params_match:
            info.model_params_b = float(params_match.group(1))
        
        # Context size: "llama_context: n_ctx         = 4096"
        ctx_match = re.search(r'llama_context: n_ctx\s+=\s+(\d+)', log_content)
        if ctx_match:
            info.context_size = int(ctx_match.group(1))
        
        # Batch size: "llama_context: n_batch       = 2048"
        batch_match = re.search(r'llama_context: n_batch\s+=\s+(\d+)', log_content)
        if batch_match:
            info.batch_size = int(batch_match.group(1))
    
    def _extract_build_info(self, log_content: str, info: HardwareInfo) -> None:
        """Extract build information from logs."""
        # Build info: "build: 7240 (61bde8e21) with Apple clang version 16.0.0"
        build_match = re.search(r'build: (\d+) \(([^)]+)\) with (.+)', log_content)
        if build_match:
            info.llama_cpp_build = build_match.group(1)
            info.llama_cpp_commit = build_match.group(2)
            info.compiler = build_match.group(3)
    
    def get_hardware_summary(self) -> Dict[str, Any]:
        """Get a concise hardware summary for display."""
        info = self.extract_hardware_info()

        summary = {}

        # GPU/Compute
        if info.gpu_name:
            summary['GPU'] = info.gpu_name
        if info.metal_backend:
            summary['Compute Backend'] = 'Metal'

        # CPU
        if info.processor:
            summary['CPU'] = info.processor

        # Memory
        if info.gpu_memory_mb:
            summary['GPU Memory'] = f"{info.gpu_memory_mb:.0f} MB"
        if info.total_memory_mb:
            summary['System Memory'] = f"{info.total_memory_mb:.0f} MB"
        if info.model_memory_mb:
            summary['Model Memory'] = f"{info.model_memory_mb:.0f} MB"

        # Model
        if info.model_size_gb:
            summary['Model Size'] = f"{info.model_size_gb:.2f} GB"
        if info.model_params_b:
            summary['Model Parameters'] = f"{info.model_params_b:.1f}B"

        # Threading
        if info.n_threads:
            summary['Threads'] = f"{info.n_threads} (batch: {info.n_threads_batch or 'N/A'})"

        # System
        if info.os_name and info.architecture:
            summary['System'] = f"{info.os_name} {info.architecture}"

        return summary


if __name__ == "__main__":
    # Test the hardware detector
    detector = HardwareDetector()
    summary = detector.get_hardware_summary()
    print("Hardware Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    info = detector.extract_hardware_info()
    print(f"\nGPU Memory: {info.gpu_memory_mb} MB" if info.gpu_memory_mb else "\nGPU Memory: Not detected")