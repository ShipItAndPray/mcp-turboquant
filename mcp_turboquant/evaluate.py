"""Perplexity evaluation for quantized models."""

from __future__ import annotations

import math
import shutil
import subprocess
from typing import Any


def evaluate_gguf(model_path: str) -> dict[str, Any]:
    """Evaluate GGUF model perplexity.

    Tries in order:
    1. llama-perplexity binary (from llama.cpp)
    2. llama-cpp-python library
    """
    # Method 1: llama-perplexity binary
    llama_perplexity = shutil.which("llama-perplexity") or shutil.which("perplexity")
    if llama_perplexity:
        try:
            cmd = [
                llama_perplexity,
                "-m",
                model_path,
                "-f",
                "wikitext-2-raw/wiki.test.raw",
                "--ctx-size",
                "512",
                "--chunks",
                "20",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "perplexity" in line.lower() and "=" in line:
                        try:
                            ppl = float(line.split("=")[-1].strip().split()[0])
                            return {
                                "success": True,
                                "perplexity": round(ppl, 2),
                                "method": "llama.cpp",
                                "dataset": "wikitext-2",
                            }
                        except (ValueError, IndexError):
                            pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # Method 2: llama-cpp-python
    try:
        from llama_cpp import Llama

        llm = Llama(model_path=model_path, n_ctx=512, verbose=False)

        test_texts = [
            "The quick brown fox jumps over the lazy dog. This is a standard test sentence used to evaluate language model quality.",
            "In machine learning, quantization refers to the process of reducing the number of bits that represent a number.",
            "The Transformer architecture has become the dominant paradigm in natural language processing and computer vision.",
            "Large language models have demonstrated remarkable capabilities in text generation and reasoning tasks.",
            "Neural networks consist of layers of interconnected nodes that process information using learned weights.",
        ]

        total_loss = 0.0
        total_tokens = 0
        for text in test_texts:
            result = llm.create_completion(
                text, max_tokens=1, logprobs=1, echo=True
            )
            if "choices" in result and result["choices"]:
                logprobs = result["choices"][0].get("logprobs", {})
                if logprobs and logprobs.get("token_logprobs"):
                    token_lps = [
                        lp
                        for lp in logprobs["token_logprobs"]
                        if lp is not None
                    ]
                    if token_lps:
                        total_loss += -sum(token_lps)
                        total_tokens += len(token_lps)

        if total_tokens > 0:
            avg_nll = total_loss / total_tokens
            ppl = math.exp(avg_nll)
            return {
                "success": True,
                "perplexity": round(ppl, 2),
                "method": "llama-cpp-python",
                "tokens_evaluated": total_tokens,
                "dataset": "built-in test passages",
            }

    except ImportError:
        pass
    except Exception as e:
        return {
            "success": False,
            "error": f"GGUF evaluation error: {e}",
        }

    return {
        "success": False,
        "error": (
            "Cannot evaluate GGUF model. "
            "Install llama-cpp-python: pip install llama-cpp-python"
        ),
        "install_cmd": "pip install llama-cpp-python",
    }


def evaluate_transformers(model_path: str, fmt: str) -> dict[str, Any]:
    """Evaluate GPTQ/AWQ model perplexity using transformers."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return {
            "success": False,
            "error": (
                "Evaluation requires transformers + torch. "
                "Install: pip install transformers torch"
            ),
            "install_cmd": "pip install transformers torch",
        }

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.float16
        )
        model.eval()

        test_texts = [
            "The quick brown fox jumps over the lazy dog. This is a standard test sentence.",
            "In machine learning, quantization reduces the number of bits that represent a number.",
            "The Transformer architecture has become the dominant paradigm in natural language processing.",
            "Large language models have demonstrated remarkable capabilities in text generation.",
            "Neural networks consist of layers of interconnected nodes that process information.",
        ]

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in test_texts:
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                num_tokens = inputs["input_ids"].shape[1]
                total_loss += loss * num_tokens
                total_tokens += num_tokens

        avg_nll = total_loss / total_tokens
        ppl = math.exp(avg_nll)

        return {
            "success": True,
            "perplexity": round(ppl, 2),
            "method": f"transformers ({fmt.upper()})",
            "tokens_evaluated": total_tokens,
            "dataset": "built-in test passages",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Evaluation error: {e}",
        }


def evaluate_model(
    model_path: str, fmt: str, bits: int
) -> dict[str, Any]:
    """Run perplexity evaluation on a quantized model.

    Args:
        model_path: Path to the quantized model file or directory.
        fmt: Format of the model ('gguf', 'gptq', or 'awq').
        bits: Bit width used for quantization.

    Returns:
        Result dict with perplexity score and quality assessment.
    """
    if fmt == "gguf":
        result = evaluate_gguf(model_path)
    elif fmt in ("gptq", "awq"):
        result = evaluate_transformers(model_path, fmt)
    else:
        return {
            "success": False,
            "error": f"Evaluation not supported for format '{fmt}'.",
        }

    # Add quality assessment if we got a perplexity score
    if result.get("success") and result.get("perplexity"):
        ppl = result["perplexity"]
        if ppl < 10:
            result["quality"] = "EXCELLENT"
            result["assessment"] = "Minimal quality loss from quantization."
        elif ppl < 20:
            result["quality"] = "GOOD"
            result["assessment"] = "Acceptable quality for most use cases."
        elif ppl < 50:
            result["quality"] = "FAIR"
            result["assessment"] = (
                f"Some quality degradation at {bits}-bit. "
                f"Consider using higher bits."
            )
        elif ppl < 100:
            result["quality"] = "DEGRADED"
            result["assessment"] = (
                f"Significant quality loss at {bits}-bit. "
                f"Recommend {min(bits + 1, 8)}-bit or higher."
            )
        else:
            result["quality"] = "POOR"
            result["assessment"] = (
                "Severe quality loss. Model may produce incoherent output. "
                "Use higher bit quantization."
            )

    result["format"] = fmt
    result["bits"] = bits
    return result
