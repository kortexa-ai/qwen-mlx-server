#!/bin/bash

# If no arguments provided, use default port 1234
if [ $# -eq 0 ]; then
    uv run python qwen_server_with_tools.py --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit --chat-template "$(cat qwen3_coder_chat_template.jinja)" --port 1234
else
    # Pass all command line arguments
    uv run python qwen_server_with_tools.py --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit --chat-template "$(cat qwen3_coder_chat_template.jinja)" "$@"
fi
