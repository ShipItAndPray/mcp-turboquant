# mcp-turboquant

The first MCP server for LLM quantization. Compress any Hugging Face model to **GGUF**, **GPTQ**, or **AWQ** format in a single tool call.

Built on [TurboQuant](https://github.com/ShipItAndPray/turboquant) — the unified CLI for model compression.

## Why?

LLM quantization is one of the most common tasks in the open-source AI workflow, yet there has been no way for AI assistants to do it autonomously. Until now.

With `mcp-turboquant`, Claude (or any MCP-compatible agent) can:

- **Quantize models** — convert any HF model to GGUF/GPTQ/AWQ with specified bit widths
- **Inspect models** — get parameter counts, architecture details, and size estimates
- **Recommend settings** — analyze your hardware and suggest optimal format + bits
- **Check backends** — verify which quantization engines are installed

## Install

### Prerequisites

```bash
pip install turboquant
```

### Claude Code

Add to your Claude Code MCP settings (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "turboquant": {
      "command": "npx",
      "args": ["-y", "mcp-turboquant"]
    }
  }
}
```

Or run locally:

```json
{
  "mcpServers": {
    "turboquant": {
      "command": "node",
      "args": ["/path/to/mcp-turboquant/index.js"]
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
      "command": "npx",
      "args": ["-y", "mcp-turboquant"]
    }
  }
}
```

## Tools

| Tool | Description |
|------|-------------|
| `quantize` | Quantize a HF model to GGUF/GPTQ/AWQ. Params: `model` (required), `format` (gguf/gptq/awq), `bits` (2-8), `output` (path) |
| `info` | Get model info — param count, architecture, size estimates |
| `recommend` | Hardware-aware recommendation for best format and bit width |
| `check` | List available quantization backends on the system |

## Examples

Once configured, just ask Claude:

> "Quantize meta-llama/Llama-3.1-8B to 4-bit GGUF"

> "What quantization format should I use for Mistral-7B on my machine?"

> "Check which quantization backends I have installed"

> "Get info on microsoft/phi-3-mini-4k-instruct"

## How It Works

`mcp-turboquant` is a lightweight Node.js process that speaks JSON-RPC over stdio (the MCP transport protocol). When an AI assistant calls a tool, the server shells out to the `turboquant` CLI and returns the output.

```
Claude / Agent  <-->  MCP Protocol (stdio)  <-->  mcp-turboquant  <-->  turboquant CLI  <-->  llama.cpp / auto-gptq / autoawq
```

No dependencies beyond Node.js and a working `turboquant` installation.

## License

MIT
