"""Quantization backends: GGUF, GPTQ, AWQ."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from typing import Any


SUPPORTED_FORMATS = ["gguf", "gptq", "awq"]
SUPPORTED_BITS = [2, 3, 4, 5, 8]

GGUF_QUANT_TYPES = {
    2: "Q2_K",
    3: "Q3_K_M",
    4: "Q4_K_M",
    5: "Q5_K_M",
    8: "Q8_0",
}


def format_size(size_bytes: int | float) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def estimate_compression(original_bits: int, target_bits: int) -> float:
    """Estimate compression ratio."""
    return original_bits / target_bits


def quantize_gguf(model_id: str, bits: int, output_dir: str) -> dict[str, Any]:
    """Quantize model to GGUF format using llama.cpp.

    Tries multiple methods in order:
    1. llama-cpp-python convert + llama-quantize binary
    2. convert_hf_to_gguf.py from llama.cpp source
    """
    quant_type = GGUF_QUANT_TYPES.get(bits, "Q4_K_M")
    output_file = os.path.join(output_dir, f"model-{quant_type}.gguf")
    os.makedirs(output_dir, exist_ok=True)

    # Method 1: Try llama-cpp-python convert + llama-quantize
    try:
        fp16_file = os.path.join(output_dir, "model-fp16.gguf")
        cmd_convert = [
            sys.executable,
            "-m",
            "llama_cpp.convert",
            "--outfile",
            fp16_file,
            "--outtype",
            "f16",
            model_id,
        ]
        result = subprocess.run(
            cmd_convert, capture_output=True, text=True, timeout=3600
        )

        if result.returncode == 0 and os.path.exists(fp16_file):
            cmd_quant = ["llama-quantize", fp16_file, output_file, quant_type]
            result = subprocess.run(
                cmd_quant, capture_output=True, text=True, timeout=3600
            )

            if result.returncode == 0 and os.path.exists(output_file):
                os.remove(fp16_file)
                return {
                    "success": True,
                    "file": output_file,
                    "size": os.path.getsize(output_file),
                    "size_human": format_size(os.path.getsize(output_file)),
                    "format": "gguf",
                    "quant_type": quant_type,
                    "bits": bits,
                }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 2: Try convert_hf_to_gguf.py from llama.cpp
    try:
        convert_script = shutil.which("convert_hf_to_gguf.py")
        if not convert_script:
            for candidate in [
                os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
                "/opt/llama.cpp/convert_hf_to_gguf.py",
            ]:
                if os.path.exists(candidate):
                    convert_script = candidate
                    break

        if convert_script:
            cmd = [
                sys.executable,
                convert_script,
                model_id,
                "--outfile",
                output_file,
                "--outtype",
                quant_type.lower(),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            )
            if result.returncode == 0 and os.path.exists(output_file):
                return {
                    "success": True,
                    "file": output_file,
                    "size": os.path.getsize(output_file),
                    "size_human": format_size(os.path.getsize(output_file)),
                    "format": "gguf",
                    "quant_type": quant_type,
                    "bits": bits,
                }
    except Exception:
        pass

    return {
        "success": False,
        "format": "gguf",
        "bits": bits,
        "error": (
            "GGUF quantization requires llama.cpp tools. "
            "Install: pip install llama-cpp-python, or build llama.cpp from source."
        ),
        "install_cmd": "pip install llama-cpp-python",
    }


def quantize_gptq(model_id: str, bits: int, output_dir: str) -> dict[str, Any]:
    """Quantize model using GPTQ via auto-gptq.

    Requires: torch, transformers, auto-gptq, datasets
    Uses c4 calibration data (128 samples, 2048 max length).
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
    except ImportError:
        return {
            "success": False,
            "format": "gptq",
            "bits": bits,
            "error": "GPTQ requires: pip install auto-gptq transformers datasets torch",
            "install_cmd": "pip install auto-gptq transformers datasets torch",
        }

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=128,
            damp_percent=0.1,
            desc_act=False,
        )

        model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)

        # Prepare calibration data from c4
        from datasets import load_dataset

        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        calibration_data = []
        for i, example in enumerate(dataset):
            if i >= 128:
                break
            tokenized = tokenizer(
                example["text"],
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            calibration_data.append(tokenized.input_ids)

        model.quantize(calibration_data)

        output_path = os.path.join(output_dir, f"model-gptq-{bits}bit")
        model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)

        total_size = sum(
            os.path.getsize(os.path.join(output_path, f))
            for f in os.listdir(output_path)
            if f.endswith((".safetensors", ".bin"))
        )

        return {
            "success": True,
            "file": output_path,
            "size": total_size,
            "size_human": format_size(total_size),
            "format": "gptq",
            "bits": bits,
            "group_size": 128,
        }

    except Exception as e:
        return {
            "success": False,
            "format": "gptq",
            "bits": bits,
            "error": str(e),
        }


def quantize_awq(model_id: str, bits: int, output_dir: str) -> dict[str, Any]:
    """Quantize model using AWQ via autoawq.

    Requires: torch, transformers, autoawq
    Uses GEMM kernel with group_size=128.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        return {
            "success": False,
            "format": "awq",
            "bits": bits,
            "error": "AWQ requires: pip install autoawq transformers torch",
            "install_cmd": "pip install autoawq transformers torch",
        }

    try:
        model = AutoAWQForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": bits,
            "version": "GEMM",
        }

        model.quantize(tokenizer, quant_config=quant_config)

        output_path = os.path.join(output_dir, f"model-awq-{bits}bit")
        model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)

        total_size = sum(
            os.path.getsize(os.path.join(output_path, f))
            for f in os.listdir(output_path)
            if f.endswith((".safetensors", ".bin"))
        )

        return {
            "success": True,
            "file": output_path,
            "size": total_size,
            "size_human": format_size(total_size),
            "format": "awq",
            "bits": bits,
            "group_size": 128,
        }

    except Exception as e:
        return {
            "success": False,
            "format": "awq",
            "bits": bits,
            "error": str(e),
        }


def quantize_model(
    model_id: str, fmt: str, bits: int, output_dir: str
) -> dict[str, Any]:
    """Dispatch quantization to the correct backend.

    Args:
        model_id: HuggingFace model ID or local path.
        fmt: One of 'gguf', 'gptq', 'awq'.
        bits: Quantization bit width (2, 3, 4, 5, or 8).
        output_dir: Directory to write output files.

    Returns:
        Result dict with success status and file info.
    """
    if fmt not in SUPPORTED_FORMATS:
        return {
            "success": False,
            "error": f"Unsupported format '{fmt}'. Use one of: {SUPPORTED_FORMATS}",
        }
    if bits not in SUPPORTED_BITS:
        return {
            "success": False,
            "error": f"Unsupported bit width {bits}. Use one of: {SUPPORTED_BITS}",
        }

    dispatch = {
        "gguf": quantize_gguf,
        "gptq": quantize_gptq,
        "awq": quantize_awq,
    }

    return dispatch[fmt](model_id, bits, output_dir)


def generate_ollama_modelfile(
    gguf_path: str, model_info: dict[str, Any], output_dir: str
) -> str:
    """Generate an Ollama Modelfile pointing to the quantized GGUF.

    Auto-detects chat template based on model architecture
    (LLaMA, Mistral, Qwen, Phi, Gemma).
    """
    arch = model_info.get("arch", "")
    context = model_info.get("context_length", 4096)

    template_str = ""
    arch_lower = arch.lower() if arch else ""

    if "llama" in arch_lower:
        template_str = textwrap.dedent("""\
            TEMPLATE \"\"\"{{- if .System }}<|start_header_id|>system<|end_header_id|>

            {{ .System }}<|eot_id|>{{- end }}
            <|start_header_id|>user<|end_header_id|>

            {{ .Prompt }}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>

            {{ .Response }}<|eot_id|>\"\"\"

            PARAMETER stop "<|eot_id|>"
            PARAMETER stop "<|end_of_text|>"
        """)
    elif "mistral" in arch_lower:
        template_str = textwrap.dedent("""\
            TEMPLATE \"\"\"[INST] {{- if .System }}{{ .System }} {{- end }}{{ .Prompt }} [/INST]{{ .Response }}\"\"\"

            PARAMETER stop "[INST]"
            PARAMETER stop "[/INST]"
        """)
    elif "qwen" in arch_lower:
        template_str = textwrap.dedent("""\
            TEMPLATE \"\"\"<|im_start|>system
            {{- if .System }}{{ .System }}{{- else }}You are a helpful assistant.{{- end }}<|im_end|>
            <|im_start|>user
            {{ .Prompt }}<|im_end|>
            <|im_start|>assistant
            {{ .Response }}<|im_end|>\"\"\"

            PARAMETER stop "<|im_start|>"
            PARAMETER stop "<|im_end|>"
        """)
    elif "phi" in arch_lower:
        template_str = textwrap.dedent("""\
            TEMPLATE \"\"\"<|system|>
            {{- if .System }}{{ .System }}{{- else }}You are a helpful assistant.{{- end }}<|end|>
            <|user|>
            {{ .Prompt }}<|end|>
            <|assistant|>
            {{ .Response }}<|end|>\"\"\"

            PARAMETER stop "<|end|>"
            PARAMETER stop "<|endoftext|>"
        """)
    elif "gemma" in arch_lower:
        template_str = textwrap.dedent("""\
            TEMPLATE \"\"\"<start_of_turn>user
            {{ .Prompt }}<end_of_turn>
            <start_of_turn>model
            {{ .Response }}<end_of_turn>\"\"\"

            PARAMETER stop "<end_of_turn>"
        """)

    gguf_filename = os.path.basename(gguf_path)
    modelfile = f"FROM ./{gguf_filename}\n\n"

    if template_str:
        modelfile += template_str + "\n"

    if context:
        modelfile += f"PARAMETER num_ctx {context}\n"

    modelfile_path = os.path.join(output_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile)

    return modelfile_path
