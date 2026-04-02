# mcp-turboquant

Self-contained Python MCP server for LLM quantization. Compress any HuggingFace model to **GGUF**, **GPTQ**, or **AWQ** format in a single tool call.

No external CLI required -- all quantization logic is embedded.

## Install

```bash
pip install mcp-turboquant
```

Or run directly with uvx:

```bash
uvx mcp-turboquant
```

### Optional backends

The `info`, `check`, and `recommend` tools work out of the box. For actual quantization, install the backend you need:

```bash
# GGUF (Ollama, llama.cpp, LM Studio)
pip install mcp-turboquant[gguf]

# GPTQ (vLLM, TGI)
pip install mcp-turboquant[gptq]

# AWQ (vLLM, TGI)
pip install mcp-turboquant[awq]

# Everything
pip install mcp-turboquant[all]
```

## Configure

### Claude Code

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "turboquant": {
      "command": "mcp-turboquant"
    }
  }
}
```

Or with uvx (no install needed):

```json
{
  "mcpServers": {
    "turboquant": {
      "command": "uvx",
      "args": ["mcp-turboquant"]
    }
  }
}
```

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "turboquant": {
      "command": "uvx",
      "args": ["mcp-turboquant"]
    }
  }
}
```

## Tools

| Tool | Description | Heavy deps? |
|------|-------------|-------------|
| `info` | Get model info from HuggingFace (params, size, architecture) | No |
| `check` | Check available quantization backends on the system | No |
| `recommend` | Hardware-aware recommendation for best format + bits | No |
| `quantize` | Quantize a model to GGUF/GPTQ/AWQ | Yes |
| `evaluate` | Run perplexity evaluation on a quantized model | Yes |
| `push` | Push quantized model to HuggingFace Hub | No |

## Examples

Once configured, ask Claude:

> "Get info on meta-llama/Llama-3.1-8B-Instruct"

> "What quantization format should I use for Mistral-7B on my machine?"

> "Quantize meta-llama/Llama-3.1-8B to 4-bit GGUF"

> "Check which quantization backends I have installed"

> "Evaluate the perplexity of my quantized model at /path/to/model.gguf"

> "Push my quantized model to myuser/model-GGUF on HuggingFace"

## How it works

```
Claude / Agent  <-->  MCP Protocol (stdio)  <-->  mcp-turboquant (Python)  <-->  llama-cpp-python / auto-gptq / autoawq
```

All quantization logic runs in-process. No external CLI tools needed.

## Run directly

```bash
# As a command
mcp-turboquant

# As a module
python -m mcp_turboquant
```

## License

MIT
