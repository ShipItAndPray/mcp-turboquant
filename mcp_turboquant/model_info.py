"""Model info retrieval and hardware-aware recommendations."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from typing import Any


def format_size(size_bytes: int | float) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def get_system_ram_gb() -> float:
    """Get system RAM in GB."""
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return round(int(result.stdout.strip()) / 1e9, 1)
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return round(kb / 1e6, 1)
    except Exception:
        pass
    return 0


def check_dependencies() -> dict[str, Any]:
    """Check which quantization backends are available."""
    import shutil

    available: dict[str, Any] = {}

    # Check for llama.cpp (GGUF)
    llama_convert = shutil.which("llama-quantize") or shutil.which("quantize")
    if llama_convert:
        available["gguf"] = True
    else:
        try:
            import llama_cpp  # noqa: F401

            available["gguf"] = True
        except ImportError:
            available["gguf"] = False

    # Check for AutoGPTQ
    try:
        import auto_gptq  # noqa: F401

        available["gptq"] = True
    except ImportError:
        available["gptq"] = False

    # Check for AutoAWQ
    try:
        import awq  # noqa: F401

        available["awq"] = True
    except ImportError:
        available["awq"] = False

    # Check for transformers
    try:
        import transformers  # noqa: F401

        available["transformers"] = True
        available["transformers_version"] = transformers.__version__
    except ImportError:
        available["transformers"] = False

    # Check for torch
    try:
        import torch

        available["torch"] = True
        available["torch_version"] = torch.__version__
        available["cuda"] = torch.cuda.is_available()
        if available["cuda"]:
            available["gpu_name"] = torch.cuda.get_device_name(0)
            available["gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / 1e9, 1
            )
        available["mps"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        available["torch"] = False
        available["cuda"] = False
        available["mps"] = False

    available["system_ram_gb"] = get_system_ram_gb()
    available["platform"] = platform.system()
    available["arch"] = platform.machine()

    return available


def get_model_info(model_id_or_path: str) -> dict[str, Any]:
    """Get model information from HuggingFace or local path."""
    info: dict[str, Any] = {"source": model_id_or_path}

    try:
        from huggingface_hub import hf_hub_download, model_info as hf_model_info

        mi = hf_model_info(model_id_or_path)
        info["model_id"] = mi.id
        info["size_bytes"] = sum(
            s.size
            for s in mi.siblings
            if s.rfilename.endswith((".safetensors", ".bin")) and s.size is not None
        )
        info["size_human"] = format_size(info["size_bytes"])

        # Try to get parameter count from config
        config_path = hf_hub_download(model_id_or_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        info["config"] = config
        info["arch"] = config.get("architectures", ["unknown"])[0]
        info["hidden_size"] = (
            config.get("hidden_size")
            or config.get("n_embd")
            or config.get("d_model")
            or 0
        )
        info["num_layers"] = (
            config.get("num_hidden_layers")
            or config.get("n_layer")
            or config.get("num_layers")
            or 0
        )
        info["vocab_size"] = config.get("vocab_size", 0)
        info["context_length"] = (
            config.get("max_position_embeddings")
            or config.get("n_positions")
            or config.get("max_seq_len")
            or config.get("seq_length")
            or 0
        )

        # Estimate parameters
        h = info["hidden_size"]
        n = info["num_layers"]
        v = info["vocab_size"]
        if h and n and v:
            params = 12 * n * h * h + v * h
            info["params_estimate"] = params
            info["params_human"] = (
                f"{params / 1e9:.1f}B" if params > 1e9 else f"{params / 1e6:.0f}M"
            )

        # If HF API didn't return file sizes, estimate from parameters
        if not info["size_bytes"] and info.get("params_estimate"):
            info["size_bytes"] = info["params_estimate"] * 2  # FP16
            info["size_human"] = format_size(info["size_bytes"]) + " (estimated)"

        info["found"] = True
    except Exception as e:
        # Check if local path
        if os.path.isdir(model_id_or_path):
            info["found"] = True
            info["local"] = True
            total = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(model_id_or_path)
                for f in fns
                if f.endswith((".safetensors", ".bin"))
            )
            info["size_bytes"] = total
            info["size_human"] = format_size(total)
            local_config = os.path.join(model_id_or_path, "config.json")
            if os.path.exists(local_config):
                with open(local_config) as f:
                    config = json.load(f)
                info["config"] = config
                info["arch"] = config.get("architectures", ["unknown"])[0]
                info["hidden_size"] = (
                    config.get("hidden_size")
                    or config.get("n_embd")
                    or config.get("d_model")
                    or 0
                )
                info["num_layers"] = (
                    config.get("num_hidden_layers")
                    or config.get("n_layer")
                    or config.get("num_layers")
                    or 0
                )
                info["vocab_size"] = config.get("vocab_size", 0)
                info["context_length"] = (
                    config.get("max_position_embeddings")
                    or config.get("n_positions")
                    or config.get("max_seq_len")
                    or 0
                )
        else:
            info["found"] = False
            info["error"] = str(e)

    return info


def recommend_format(
    model_info: dict[str, Any], deps: dict[str, Any]
) -> list[dict[str, Any]]:
    """Recommend the best quantization format based on hardware and model."""
    model_size_gb = model_info.get("size_bytes", 0) / 1e9
    params = model_info.get("params_estimate", 0)
    params_b = params / 1e9 if params else 0

    has_cuda = deps.get("cuda", False)
    gpu_name = deps.get("gpu_name", "")
    gpu_mem = deps.get("gpu_mem_gb", 0)
    has_mps = deps.get("mps", False)
    system_ram = deps.get("system_ram_gb", 0) or get_system_ram_gb()

    hardware = {}
    if has_cuda:
        hardware["accelerator"] = f"CUDA GPU: {gpu_name} ({gpu_mem}GB VRAM)"
    elif has_mps:
        hardware["accelerator"] = f"Apple Silicon (MPS) — {system_ram}GB unified memory"
    else:
        hardware["accelerator"] = "None (CPU only)"
    hardware["ram_gb"] = system_ram

    recommendations: list[dict[str, Any]] = []

    # Estimate quantized sizes
    size_4bit = model_size_gb / 4 if model_size_gb else params_b * 0.5
    size_8bit = model_size_gb / 2 if model_size_gb else params_b * 1.0

    source = model_info.get("source", "MODEL")

    def _make_rec(rank, label, fmt, bits, reason, use_case):
        return {
            "rank": rank,
            "label": label,
            "format": fmt,
            "bits": bits,
            "reason": reason,
            "use_case": use_case,
            "command": f'quantize(model="{source}", format="{fmt.lower()}", bits={bits})',
        }

    if has_cuda and gpu_mem > 0:
        if size_4bit * 1.2 <= gpu_mem:
            recommendations.append(_make_rec(
                1, "BEST", "AWQ", 4,
                f"Best GPU throughput. 4-bit model (~{size_4bit:.1f}GB) fits in {gpu_mem}GB VRAM.",
                "Production GPU serving with vLLM or TGI",
            ))
            recommendations.append(_make_rec(
                2, "ALSO GOOD", "GPTQ", 4,
                "Alternative GPU format. Wider tool support than AWQ.",
                "GPU serving when AWQ isn't available",
            ))
            recommendations.append(_make_rec(
                3, "ALTERNATIVE", "GGUF", 4,
                "Universal format. Works with Ollama, LM Studio, llama.cpp.",
                "Local use, sharing, or CPU fallback",
            ))
        elif size_4bit * 1.2 > gpu_mem and size_4bit <= system_ram:
            recommendations.append(_make_rec(
                1, "BEST", "GGUF", 4,
                f"Model too large for {gpu_mem}GB VRAM. GGUF supports CPU+GPU split.",
                "CPU+GPU hybrid inference via llama.cpp",
            ))
            if params_b > 13:
                recommendations.append(_make_rec(
                    2, "ALSO GOOD", "GGUF", 2,
                    f"Aggressive compression to fit in {gpu_mem}GB VRAM. Quality trade-off.",
                    "When VRAM is tight and you need GPU acceleration",
                ))
        else:
            recommendations.append(_make_rec(
                1, "BEST", "GGUF", 2,
                "Model requires aggressive compression for your hardware.",
                "Maximum compression for large models",
            ))

    elif has_mps:
        recommendations.append(_make_rec(
            1, "BEST", "GGUF", 4,
            "Best format for Apple Silicon. llama.cpp has Metal acceleration.",
            "Ollama or LM Studio on Mac",
        ))
        if size_8bit <= system_ram * 0.7:
            recommendations.append(_make_rec(
                2, "ALSO GOOD", "GGUF", 8,
                f"Higher quality, still fits in {system_ram}GB unified memory.",
                "Maximum quality on Mac",
            ))

    else:
        recommendations.append(_make_rec(
            1, "BEST", "GGUF", 4,
            "Only format that runs well on CPU. Use with Ollama or llama.cpp.",
            "CPU inference via Ollama or llama.cpp",
        ))
        if params_b <= 3 and size_8bit <= system_ram * 0.5:
            recommendations.append(_make_rec(
                2, "ALSO GOOD", "GGUF", 8,
                f"Small model ({model_info.get('params_human', '')}). Higher quality fits in RAM.",
                "Better quality for small models on CPU",
            ))

    return {
        "model": source,
        "model_params": model_info.get("params_human", "unknown"),
        "model_size": model_info.get("size_human", "unknown"),
        "hardware": hardware,
        "recommendations": recommendations,
    }
