# Qwen MLX Server with Tool Support

An enhanced MLX server that provides OpenAI-compatible tool calling for Qwen3 models by parsing their native XML format and converting it to OpenAI JSON format.

## Features

- ✅ **XML to JSON conversion**: Automatically converts Qwen3's `<tool_call>` XML format to OpenAI JSON
- ✅ **OpenAI compatibility**: Drop-in replacement for OpenAI's chat completions API
- ✅ **Streaming support**: Proper streaming with XML filtering to prevent raw XML in output
- ✅ **Robust parsing**: Handles incomplete and malformed XML gracefully
- ✅ **vLLM compliance**: Based on official vLLM Qwen3XMLToolParser implementation

## Installation

```bash
git clone https://github.com/yourusername/qwen-mlx-server.git
cd qwen-mlx-server
pip install -r requirements.txt
```

## Quick Start

```bash
# Basic usage with default template
python qwen_server_with_tools.py --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit

# With custom chat template for better tool calling
python qwen_server_with_tools.py \
  --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
  --chat-template "$(cat qwen3_coder_chat_template.jinja)"

# With different log level (WARNING for production, DEBUG for development)
python qwen_server_with_tools.py \
  --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
  --log-level WARNING

# With existing LM Studio download
python qwen_server_with_tools.py \
  --model ~/.cache/lm-studio/models/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
  --chat-template "$(cat qwen3_coder_chat_template.jinja)"
# Note, when entering API details into a tool such as Qwen Code, the model name should be "default_model"
# to avoid a redownload of the model.
```

## Usage Example

### Tool Calling Request

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
    "messages": [
      {
        "role": "user",
        "content": "Calculate 15 * 7"
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "calculate",
          "description": "Perform mathematical calculations",
          "parameters": {
            "type": "object",
            "properties": {
              "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate"
              }
            },
            "required": ["expression"]
          }
        }
      }
    ],
    "stream": false
  }'
```

### How It Works

The server automatically converts Qwen3's native XML output:
```xml
<tool_call>
<function=calculate>
<parameter=expression>
15 * 7
</parameter>
</function>
</tool_call>
```

To OpenAI-compatible JSON:
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "",
      "tool_calls": [{
        "type": "function",
        "id": "call_12345",
        "function": {
          "name": "calculate",
          "arguments": "{\"expression\": \"15 * 7\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | HuggingFace model path | `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` |
| `--host` | Server host | `127.0.0.1` |
| `--port` | Server port | `8080` |
| `--chat-template` | Custom chat template file | `""` (uses model default) |
| `--use-default-chat-template` | Force use of model's default template | `False` |
| `--log-level` | Logging verbosity | `INFO` |
| `--max-tokens` | Default max tokens to generate | `512` |

## Logging Levels

- `DEBUG`: Shows detailed XML parsing and conversion steps (useful for development)
- `INFO`: Standard operational messages (default)
- `WARNING`: Only warnings and errors (recommended for production)
- `ERROR`: Only errors

## Files

- `qwen_server_with_tools.py` - Main server with XML→JSON tool parsing
- `qwen3_coder_chat_template.jinja` - Optimized Qwen3-Coder chat template
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## Supported Models

Designed for Qwen3-Coder models but should work with any Qwen model that outputs XML tool calls:

- `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`
- `mlx-community/Qwen3-Coder-7B-A3B-Instruct-4bit`
- Other Qwen3 variants

## Development

To see detailed XML parsing logs:
```bash
python qwen_server_with_tools.py \
  --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
  --log-level DEBUG
```

## Implementation Notes

- Based on vLLM's `Qwen3XMLToolParser` for maximum compatibility
- Handles both streaming and non-streaming requests correctly  
- Gracefully handles incomplete XML during token-by-token generation
- Maintains full OpenAI Chat Completions API compatibility
- Supports parameter type conversion and validation
- Filters XML from streaming output to prevent malformed responses

## License

MIT License - feel free to use this in your projects!
