"""MCP TurboQuant server — LLM quantization tools via MCP protocol."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from mcp_turboquant import __version__
from mcp_turboquant.evaluate import evaluate_model
from mcp_turboquant.model_info import (
    check_dependencies,
    format_size,
    get_model_info,
    recommend_format,
)
from mcp_turboquant.quantize import (
    SUPPORTED_BITS,
    SUPPORTED_FORMATS,
    estimate_compression,
    generate_ollama_modelfile,
    quantize_model,
)

mcp = FastMCP(
    "TurboQuant",
    instructions=(
        "LLM quantization MCP server. Compress any HuggingFace model "
        "to GGUF, GPTQ, or AWQ format. Version: " + __version__
    ),
)


@mcp.tool()
def info(model: str) -> dict[str, Any]:
    """Get model info from HuggingFace — parameters, size, architecture.

    Lightweight call using the HuggingFace API. No GPU or heavy
    dependencies required.

    Args:
        model: HuggingFace model ID (e.g. 'meta-llama/Llama-3.1-8B-Instruct')
               or local path to a model directory.

    Returns:
        Model metadata including architecture, parameter count, size,
        hidden dimensions, number of layers, vocabulary size, and
        context length.
    """
    result = get_model_info(model)

    if not result.get("found"):
        return {
            "error": f"Model not found: {result.get('error', 'unknown')}",
            "model": model,
        }

    # Build a clean response (strip internal fields like raw config)
    output = {
        "model": result.get("model_id", result.get("source")),
        "found": True,
        "architecture": result.get("arch", "unknown"),
        "parameters": result.get("params_human", "unknown"),
        "size": result.get("size_human", "unknown"),
        "size_bytes": result.get("size_bytes", 0),
        "hidden_size": result.get("hidden_size", 0),
        "num_layers": result.get("num_layers", 0),
        "vocab_size": result.get("vocab_size", 0),
        "context_length": result.get("context_length", 0),
    }

    if result.get("local"):
        output["local"] = True

    # Add compression estimates
    if result.get("size_bytes"):
        sz = result["size_bytes"]
        output["estimated_sizes"] = {
            "4bit": format_size(sz / estimate_compression(16, 4)),
            "8bit": format_size(sz / estimate_compression(16, 8)),
        }

    return output


@mcp.tool()
def check() -> dict[str, Any]:
    """Check available quantization backends on this system.

    Reports which quantization engines (GGUF/GPTQ/AWQ) are installed,
    whether PyTorch and transformers are available, GPU information
    (CUDA or Apple MPS), and system RAM.

    No arguments required. Lightweight system check.

    Returns:
        Dictionary of available backends and hardware info.
    """
    deps = check_dependencies()

    backends = {
        "gguf": {
            "available": deps.get("gguf", False),
            "install": "pip install llama-cpp-python",
        },
        "gptq": {
            "available": deps.get("gptq", False),
            "install": "pip install auto-gptq datasets",
        },
        "awq": {
            "available": deps.get("awq", False),
            "install": "pip install autoawq",
        },
    }

    hardware = {
        "platform": deps.get("platform", "unknown"),
        "arch": deps.get("arch", "unknown"),
        "system_ram_gb": deps.get("system_ram_gb", 0),
    }

    if deps.get("cuda"):
        hardware["gpu"] = deps.get("gpu_name", "CUDA GPU")
        hardware["gpu_mem_gb"] = deps.get("gpu_mem_gb", 0)
        hardware["accelerator"] = "cuda"
    elif deps.get("mps"):
        hardware["accelerator"] = "mps"
        hardware["gpu"] = "Apple Silicon (Metal Performance Shaders)"
    else:
        hardware["accelerator"] = "cpu"

    core = {
        "torch": {
            "available": deps.get("torch", False),
            "version": deps.get("torch_version", None),
            "install": "pip install torch",
        },
        "transformers": {
            "available": deps.get("transformers", False),
            "version": deps.get("transformers_version", None),
            "install": "pip install transformers",
        },
    }

    return {
        "backends": backends,
        "core_dependencies": core,
        "hardware": hardware,
        "server_version": __version__,
    }


@mcp.tool()
def recommend(model: str) -> dict[str, Any]:
    """Recommend best quantization format and bit width for a model.

    Analyzes the model size and your hardware (GPU VRAM, Apple Silicon,
    system RAM) to suggest the optimal format (GGUF/GPTQ/AWQ) and bit
    width (2-8). Ranked recommendations with use-case explanations.

    Args:
        model: HuggingFace model ID (e.g. 'meta-llama/Llama-3.1-8B-Instruct')
               or local path to a model directory.

    Returns:
        Ranked recommendations with format, bits, reasoning, and use cases.
    """
    model_info = get_model_info(model)

    if not model_info.get("found"):
        return {
            "error": f"Model not found: {model_info.get('error', 'unknown')}",
            "model": model,
        }

    deps = check_dependencies()
    return recommend_format(model_info, deps)


@mcp.tool()
def quantize(
    model: str,
    format: Literal["gguf", "gptq", "awq"] = "gguf",
    bits: Literal[2, 3, 4, 5, 8] = 4,
    output_dir: str | None = None,
    target: Literal["ollama", "vllm", "llamacpp", "lmstudio"] | None = None,
) -> dict[str, Any]:
    """Quantize a HuggingFace model to GGUF, GPTQ, or AWQ format.

    This is a heavy operation that downloads and compresses the model.
    Requires appropriate backend dependencies to be installed.

    Args:
        model: HuggingFace model ID (e.g. 'meta-llama/Llama-3.1-8B-Instruct')
               or local path to a model directory.
        format: Output format — gguf, gptq, or awq. Default: gguf.
        bits: Quantization bit width — 2, 3, 4, 5, or 8. Default: 4.
        output_dir: Directory to write output files. Default: temp directory.
        target: Deployment target. ollama/llamacpp/lmstudio force GGUF, vllm forces AWQ.

    Returns:
        Quantization result with file paths, sizes, and compression ratios.
    """
    # Resolve target overrides
    fmt = format.lower()
    if target:
        target = target.lower()
        if target == "ollama":
            fmt = "gguf"
        elif target == "vllm":
            fmt = "awq"
        elif target in ("llamacpp", "lmstudio"):
            fmt = "gguf"

    if fmt not in SUPPORTED_FORMATS:
        return {
            "error": f"Unsupported format '{fmt}'. Use one of: {SUPPORTED_FORMATS}",
        }
    if bits not in SUPPORTED_BITS:
        return {
            "error": f"Unsupported bit width {bits}. Use one of: {SUPPORTED_BITS}",
        }

    # Get model info for the report
    model_info = get_model_info(model)
    if not model_info.get("found"):
        return {
            "error": f"Model not found: {model_info.get('error', 'unknown')}",
            "model": model,
        }

    # Set up output directory
    if not output_dir:
        model_slug = model.replace("/", "-").replace(".", "-")
        output_dir = os.path.join(
            tempfile.gettempdir(), "turboquant", f"{model_slug}-{fmt}-{bits}bit"
        )
    os.makedirs(output_dir, exist_ok=True)

    # Run quantization
    result = quantize_model(model, fmt, bits, output_dir)

    # Build response
    response = {
        "model": model,
        "architecture": model_info.get("arch", "unknown"),
        "parameters": model_info.get("params_human", "unknown"),
        "original_size": model_info.get("size_human", "unknown"),
        "target_bits": bits,
        "format": fmt,
        "theoretical_compression": f"{estimate_compression(16, bits):.1f}x",
    }

    if result["success"]:
        response["success"] = True
        response["output_file"] = result["file"]
        response["output_size"] = result.get("size_human", "unknown")
        response["output_size_bytes"] = result.get("size", 0)

        original_bytes = model_info.get("size_bytes", 0)
        if original_bytes and result.get("size"):
            actual = original_bytes / result["size"]
            response["actual_compression"] = f"{actual:.1f}x"

        if result.get("quant_type"):
            response["quant_type"] = result["quant_type"]

        # Generate Ollama Modelfile if target is ollama
        if target == "ollama" and fmt == "gguf":
            modelfile_path = generate_ollama_modelfile(
                result["file"], model_info, output_dir
            )
            model_name = model.split("/")[-1].lower().replace(".", "-")
            quant_type = result.get("quant_type", "Q4_K_M")
            response["ollama"] = {
                "modelfile": modelfile_path,
                "import_command": f"cd {output_dir} && ollama create {model_name}-{quant_type.lower()} -f Modelfile",
                "run_command": f"ollama run {model_name}-{quant_type.lower()}",
            }
    else:
        response["success"] = False
        response["error"] = result.get("error", "Unknown error")
        if result.get("install_cmd"):
            response["install_cmd"] = result["install_cmd"]

    return response


@mcp.tool()
def evaluate(
    model_path: str,
    format: str = "gguf",
    bits: int = 4,
) -> dict[str, Any]:
    """Run perplexity evaluation on a quantized model.

    Measures model quality after quantization using perplexity scoring.
    Lower perplexity = better quality. Includes a quality assessment
    (EXCELLENT/GOOD/FAIR/DEGRADED/POOR).

    Args:
        model_path: Path to the quantized model file (GGUF) or directory
                    (GPTQ/AWQ).
        format: Format of the quantized model. One of 'gguf', 'gptq', 'awq'.
        bits: Bit width used during quantization (for quality context).

    Returns:
        Perplexity score, quality assessment, and evaluation metadata.
    """
    if not os.path.exists(model_path):
        return {
            "success": False,
            "error": f"Model path does not exist: {model_path}",
        }

    return evaluate_model(model_path, format.lower(), bits)


@mcp.tool()
def push(
    repo_id: str,
    model_dir: str,
    model: str | None = None,
    bits: int = 4,
) -> dict[str, Any]:
    """Push a quantized model to HuggingFace Hub.

    Uploads all model files from the output directory to a HuggingFace
    repository. Generates a model card (README.md) with metadata.
    Requires HuggingFace authentication (huggingface-cli login or HF_TOKEN).

    Args:
        repo_id: HuggingFace repository ID (e.g. 'username/model-GGUF-4bit').
        model_dir: Local directory containing the quantized model files.
        model: Original model ID for the model card (optional).
        bits: Bit width used during quantization (for model card metadata).

    Returns:
        Upload result with repository URL and file count.
    """
    if not os.path.isdir(model_dir):
        return {
            "success": False,
            "error": f"Directory does not exist: {model_dir}",
        }

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return {
            "success": False,
            "error": "huggingface-hub required. Install: pip install huggingface-hub",
            "install_cmd": "pip install huggingface-hub",
        }

    api = HfApi()

    # Check authentication
    try:
        user_info = api.whoami()
        username = user_info.get("name", "unknown")
    except Exception:
        return {
            "success": False,
            "error": (
                "Not authenticated with HuggingFace. "
                "Run: huggingface-cli login, or set HF_TOKEN environment variable."
            ),
        }

    # Create repo if needed
    try:
        api.create_repo(repo_id, exist_ok=True, repo_type="model")
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to create repository: {e}",
        }

    # Generate model card
    model_source = model or "unknown"
    card = _generate_model_card(model_source, repo_id, bits)
    card_path = os.path.join(model_dir, "README.md")
    with open(card_path, "w") as f:
        f.write(card)

    # Upload all files
    files_uploaded = 0
    errors = []
    for root, _dirs, files in os.walk(model_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, model_dir)
            try:
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=rel_path,
                    repo_id=repo_id,
                    repo_type="model",
                )
                files_uploaded += 1
            except Exception as e:
                errors.append(f"{rel_path}: {e}")

    result = {
        "success": files_uploaded > 0,
        "repository": f"https://huggingface.co/{repo_id}",
        "files_uploaded": files_uploaded,
        "authenticated_as": username,
    }

    if errors:
        result["upload_errors"] = errors

    return result


def _generate_model_card(model_id: str, hub_repo: str, bits: int) -> str:
    """Generate a HuggingFace model card for uploaded quantized models."""
    return f"""---
base_model: {model_id}
tags:
- quantized
- turboquant
- {bits}bit
license: mit
---

# {hub_repo.split('/')[-1]}

**Quantized version of [{model_id}](https://huggingface.co/{model_id})**

Quantized with [TurboQuant MCP](https://github.com/ShipItAndPray/mcp-turboquant) -- compress any LLM via MCP tools.

## Details

| Property | Value |
|----------|-------|
| Base Model | [{model_id}](https://huggingface.co/{model_id}) |
| Quantization | {bits}-bit |

## Usage

### GGUF (Ollama / llama.cpp / LM Studio)

```bash
huggingface-cli download {hub_repo} --include "*.gguf"
```

### GPTQ / AWQ (vLLM / TGI)

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("{hub_repo}")
```

---
*Quantized with [mcp-turboquant](https://github.com/ShipItAndPray/mcp-turboquant)*
"""


def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
