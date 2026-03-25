#!/usr/bin/env node
"use strict";

const { execSync } = require("child_process");
const readline = require("readline");

const TOOLS = [
  {
    name: "quantize",
    description:
      "Quantize a Hugging Face model to GGUF, GPTQ, or AWQ format. Returns the path to the quantized model.",
    inputSchema: {
      type: "object",
      properties: {
        model: {
          type: "string",
          description:
            "Hugging Face model ID or local path (e.g. meta-llama/Llama-3.1-8B)",
        },
        format: {
          type: "string",
          enum: ["gguf", "gptq", "awq"],
          description: "Quantization format (default: gguf)",
          default: "gguf",
        },
        bits: {
          type: "number",
          enum: [2, 3, 4, 5, 6, 8],
          description: "Bit width for quantization (default: 4)",
          default: 4,
        },
        output: {
          type: "string",
          description:
            "Output directory or file path (optional, defaults to turboquant default)",
        },
      },
      required: ["model"],
    },
  },
  {
    name: "info",
    description:
      "Get detailed information about a model — parameter count, architecture, size, and recommended quantization settings.",
    inputSchema: {
      type: "object",
      properties: {
        model: {
          type: "string",
          description: "Hugging Face model ID or local path",
        },
      },
      required: ["model"],
    },
  },
  {
    name: "recommend",
    description:
      "Recommend the best quantization format and bit width for your hardware. Analyzes available VRAM, RAM, and compute capability.",
    inputSchema: {
      type: "object",
      properties: {
        model: {
          type: "string",
          description: "Hugging Face model ID or local path",
        },
      },
      required: ["model"],
    },
  },
  {
    name: "check",
    description:
      "Check which quantization backends are available on this system (llama.cpp, auto-gptq, autoawq, etc.).",
    inputSchema: {
      type: "object",
      properties: {},
    },
  },
];

function runCommand(cmd, timeoutMs = 600000) {
  try {
    const output = execSync(cmd, {
      encoding: "utf-8",
      timeout: timeoutMs,
      maxBuffer: 50 * 1024 * 1024,
      stdio: ["pipe", "pipe", "pipe"],
    });
    return { success: true, output: output.trim() };
  } catch (err) {
    const stderr = err.stderr ? err.stderr.toString().trim() : "";
    const stdout = err.stdout ? err.stdout.toString().trim() : "";
    return {
      success: false,
      output: stderr || stdout || err.message,
    };
  }
}

function handleToolCall(name, args) {
  switch (name) {
    case "quantize": {
      const model = args.model;
      const format = args.format || "gguf";
      const bits = args.bits || 4;
      let cmd = `turboquant "${model}" --format ${format} --bits ${bits}`;
      if (args.output) cmd += ` --output "${args.output}"`;
      const result = runCommand(cmd);
      return {
        content: [
          {
            type: "text",
            text: result.success
              ? `Quantization complete.\n\n${result.output}`
              : `Quantization failed.\n\n${result.output}`,
          },
        ],
        isError: !result.success,
      };
    }
    case "info": {
      const result = runCommand(`turboquant "${args.model}" --info`);
      return {
        content: [{ type: "text", text: result.output }],
        isError: !result.success,
      };
    }
    case "recommend": {
      const result = runCommand(`turboquant "${args.model}" --recommend`);
      return {
        content: [{ type: "text", text: result.output }],
        isError: !result.success,
      };
    }
    case "check": {
      const result = runCommand("turboquant --check");
      return {
        content: [{ type: "text", text: result.output }],
        isError: !result.success,
      };
    }
    default:
      return {
        content: [{ type: "text", text: `Unknown tool: ${name}` }],
        isError: true,
      };
  }
}

function makeResponse(id, result) {
  return JSON.stringify({ jsonrpc: "2.0", id, result });
}

function makeError(id, code, message) {
  return JSON.stringify({ jsonrpc: "2.0", id, error: { code, message } });
}

// JSON-RPC over stdio
const rl = readline.createInterface({ input: process.stdin, terminal: false });
let buffer = "";

rl.on("line", (line) => {
  buffer += line;
  let msg;
  try {
    msg = JSON.parse(buffer);
    buffer = "";
  } catch {
    return; // incomplete JSON, wait for more
  }

  const { id, method, params } = msg;

  switch (method) {
    case "initialize":
      process.stdout.write(
        makeResponse(id, {
          protocolVersion: "2024-11-05",
          capabilities: { tools: {} },
          serverInfo: {
            name: "mcp-turboquant",
            version: "0.1.0",
          },
        }) + "\n"
      );
      break;

    case "notifications/initialized":
      // no response needed for notifications
      break;

    case "tools/list":
      process.stdout.write(
        makeResponse(id, { tools: TOOLS }) + "\n"
      );
      break;

    case "tools/call": {
      const toolName = params?.name;
      const toolArgs = params?.arguments || {};
      const result = handleToolCall(toolName, toolArgs);
      process.stdout.write(makeResponse(id, result) + "\n");
      break;
    }

    default:
      if (id !== undefined) {
        process.stdout.write(
          makeError(id, -32601, `Method not found: ${method}`) + "\n"
        );
      }
  }
});

process.stdin.on("end", () => process.exit(0));
