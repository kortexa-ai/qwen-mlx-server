#!/usr/bin/env python3

"""
MLX Server with enhanced Qwen3 XML tool parsing support.

This script extends the MLX server to properly parse Qwen3's XML-style tool calls
and convert them to OpenAI-compatible JSON format responses.

Based on vLLM's Qwen3XMLToolParser implementation.
"""

import json
import re
import uuid
import argparse
import logging
from typing import Dict, List, Optional, Any, Union
from collections.abc import Sequence

# Import MLX server components
from mlx_lm.server import APIHandler, ModelProvider, PromptCache, run
from mlx_lm.utils import load


class Qwen3ToolParser:
    """Tool parser for Qwen3's XML format that converts to OpenAI JSON format."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # Track tool calls for finish_reason handling (like vLLM)
        self.prev_tool_call_arr = []
        
        # XML parsing patterns (matching vLLM exactly)
        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)</parameter>|<parameter=(.*?)$", re.DOTALL
        )
        
        # Sentinel tokens
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_prefix = "<function="
        
    def _convert_param_value(self, param_value: str, param_name: str, param_config: dict, func_name: str) -> Any:
        """Convert parameter value based on its expected type."""
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logging.warning(
                    f"Parsed parameter '{param_name}' is not defined in the tool "
                    f"parameters for tool '{func_name}', directly returning the string value."
                )
            return param_value

        if (
            isinstance(param_config[param_name], dict)
            and "type" in param_config[param_name]
        ):
            param_type = str(param_config[param_name]["type"]).strip().lower()
        else:
            param_type = "string"
            
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                param_value = int(param_value)
            except:
                logging.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not an integer in tool "
                    f"'{func_name}', degenerating to string."
                )
            return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_param_value = float(param_value)
                param_value = float_param_value if float_param_value - int(float_param_value) != 0 else int(float_param_value)
            except:
                logging.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a float in tool "
                    f"'{func_name}', degenerating to string."
                )
            return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logging.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a boolean (`true` of `false`) in tool '{func_name}', degenerating to false."
                )
            return param_value == "true"
        else:
            if param_type == "object" or param_type.startswith("dict"):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except:
                    logging.warning(
                        f"Parsed value '{param_value}' of parameter '{param_name}' is not a valid JSON object in tool "
                        f"'{func_name}', will try other methods to parse it."
                    )
            try:
                param_value = eval(param_value)
            except:
                logging.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' cannot be converted via Python `eval()` in tool '{func_name}', degenerating to string."
                )
            return param_value

    def _get_arguments_config(self, func_name: str, tools: Optional[List[Dict]]) -> dict:
        """Get parameter configuration for a function from tools list."""
        if tools is None:
            return {}
        for config in tools:
            if not isinstance(config, dict):
                continue
            if config.get("type") == "function" and isinstance(config.get("function"), dict):
                if config["function"].get("name") == func_name:
                    params = config["function"].get("parameters", {})
                    if isinstance(params, dict) and "properties" in params:
                        return params["properties"]
                    elif isinstance(params, dict):
                        return params
                    else:
                        return {}
        logging.warning(f"Tool '{func_name}' is not defined in the tools list.")
        return {}

    def _parse_xml_function_call(self, function_call_str: str, tools: Optional[List[Dict]]) -> Optional[Dict]:
        """Parse XML function call format to OpenAI JSON format."""
        try:
            # Handle incomplete XML gracefully
            if ">" not in function_call_str:
                logging.warning(f"Incomplete XML function call: {function_call_str[:100]}...")
                return None
                
            # Extract function name
            end_index = function_call_str.index(">")
            function_name = function_call_str[:end_index]
            param_config = self._get_arguments_config(function_name, tools)
            parameters = function_call_str[end_index + 1:]
            
            param_dict = {}
            
            # Handle incomplete parameters more gracefully
            parameter_matches = self.tool_call_parameter_regex.findall(parameters)
            for match in parameter_matches:
                try:
                    match_text = match[0] if match[0] else match[1]
                    if ">" not in match_text:
                        logging.warning(f"Incomplete parameter in XML: {match_text[:50]}...")
                        continue
                        
                    idx = match_text.index(">")
                    param_name = match_text[:idx]
                    param_value = str(match_text[idx + 1:])
                    
                    # Remove prefix and trailing \n
                    if param_value.startswith("\n"):
                        param_value = param_value[1:]
                    if param_value.endswith("\n"):
                        param_value = param_value[:-1]

                    param_dict[param_name] = self._convert_param_value(
                        param_value, param_name, param_config, function_name
                    )
                except Exception as param_e:
                    logging.warning(f"Error parsing parameter {match}: {param_e}")
                    continue
                
            return {
                "type": "function",
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(param_dict, ensure_ascii=False)
                }
            }
        except Exception as e:
            logging.error(f"Error parsing XML function call '{function_call_str[:100]}...': {e}")
            return None

    def _get_function_calls(self, model_output: str) -> List[str]:
        """Extract function calls from model output (matching vLLM implementation)."""
        # Find all tool calls
        matched_ranges = self.tool_call_regex.findall(model_output)
        raw_tool_calls = [
            match[0] if match[0] else match[1] for match in matched_ranges
        ]

        # Back-off strategy if no tool_call tags found (like vLLM)
        if len(raw_tool_calls) == 0:
            raw_tool_calls = [model_output]

        raw_function_calls = []
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(self.tool_call_function_regex.findall(tool_call))

        function_calls = [
            match[0] if match[0] else match[1] for match in raw_function_calls
        ]
        return function_calls

    def extract_tool_calls(self, model_output: str, tools: Optional[List[Dict]] = None) -> Dict:
        """Extract tool calls from model output and return in OpenAI format."""
        # Quick check to avoid unnecessary processing (like vLLM)
        if self.tool_call_prefix not in model_output:
            return {
                "tools_called": False,
                "tool_calls": [],
                "content": model_output
            }

        try:
            function_calls = self._get_function_calls(model_output)
            if len(function_calls) == 0:
                return {
                    "tools_called": False,
                    "tool_calls": [],
                    "content": model_output
                }

            tool_calls = []
            for function_call_str in function_calls:
                parsed_call = self._parse_xml_function_call(function_call_str, tools)
                if parsed_call:
                    tool_calls.append(parsed_call)

            # Populate prev_tool_call_arr for serving layer to set finish_reason (like vLLM)
            self.prev_tool_call_arr.clear()  # Clear previous calls
            for tool_call in tool_calls:
                if tool_call:
                    self.prev_tool_call_arr.append({
                        "name": tool_call["function"]["name"],
                        "arguments": tool_call["function"]["arguments"],
                    })

            # Extract content before tool calls (like vLLM - no rstrip)
            content_index = model_output.find(self.tool_call_start_token)
            content_index = (
                content_index
                if content_index >= 0
                else model_output.find(self.tool_call_prefix)
            )
            content = model_output[:content_index] if content_index > 0 else ""

            return {
                "tools_called": len(tool_calls) > 0,
                "tool_calls": tool_calls,
                "content": content if content else None
            }

        except Exception as e:
            logging.error(f"Error in extracting tool call from response: {e}")
            return {
                "tools_called": False,
                "tool_calls": [],
                "content": model_output
            }


class EnhancedAPIHandler(APIHandler):
    """Enhanced API handler with Qwen3 XML tool parsing support."""
    
    def __init__(
        self,
        model_provider: ModelProvider,
        *args,
        prompt_cache: Optional[PromptCache] = None,
        system_fingerprint: Optional[str] = None,
        **kwargs,
    ):
        self.tool_parser = None
        self.full_generated_text = ""  # Store full text including tool calls
        self.in_tool_call = False  # Track if we're currently in a tool call during streaming
        self.tool_text = ""  # For original MLX server compatibility
        super().__init__(model_provider, *args, prompt_cache=prompt_cache, system_fingerprint=system_fingerprint, **kwargs)
        # Initialize tool parser after parent initialization
        if hasattr(self, 'tokenizer'):
            self.tool_parser = Qwen3ToolParser(self.tokenizer)
    
    def handle_completion(self, prompt, stop_id_sequences):
        """Override to capture full generated text including XML tool calls."""
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors
        import mlx.core as mx
        import time
        
        # Reset streaming state for new request
        self.in_tool_call = False
        self.tool_text = ""
        
        tokens = []
        finish_reason = "length"
        stop_sequence_suffix = None
        if self.stream:
            self.end_headers()
            logging.debug(f"Starting stream:")
        else:
            logging.debug(f"Starting completion:")
        token_logprobs = []
        top_tokens = []

        # Debug: Log the request body
        logging.debug(f"REQUEST BODY: {json.dumps(self.body, indent=2)}")
        
        prompt = self.get_prompt_cache(prompt)

        text = ""
        full_text = ""  # Capture ALL generated text including XML
        tic = time.perf_counter()
        sampler = make_sampler(
            self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            xtc_probability=self.xtc_probability,
            xtc_threshold=self.xtc_threshold,
            xtc_special_tokens=[
                self.tokenizer.eos_token_id,
                self.tokenizer.encode("\n"),
            ],
        )
        logits_processors = make_logits_processors(
            self.logit_bias,
            self.repetition_penalty,
            self.repetition_context_size,
        )

        tool_calls = []
        segment = ""
        
        for gen_response in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt_cache=self.prompt_cache.cache,
            draft_model=self.model_provider.draft_model,
            num_draft_tokens=self.num_draft_tokens,
        ):
            logging.debug(gen_response.text)
            
            # Capture ALL text for XML parsing
            full_text += gen_response.text
            
            # Use original logic for tool calling detection
            # For Qwen models, tokenizer.has_tool_calling is False, so all text goes to 'text'
            if (
                self.tokenizer.has_tool_calling
                and gen_response.text == self.tokenizer.tool_call_start
            ):
                in_tool_call = True
            elif hasattr(self, 'in_tool_call') and self.in_tool_call:
                if gen_response.text == self.tokenizer.tool_call_end:
                    tool_calls.append(self.tool_text)
                    self.tool_text = ""
                    self.in_tool_call = False
                else:
                    self.tool_text += gen_response.text
            else:
                text += gen_response.text
                segment += gen_response.text
                
            token = gen_response.token
            logprobs = gen_response.logprobs
            tokens.append(token)

            if self.logprobs > 0:
                sorted_indices = mx.argpartition(-logprobs, kth=self.logprobs - 1)
                top_indices = sorted_indices[: self.logprobs]
                top_logprobs = logprobs[top_indices]
                top_token_info = zip(top_indices.tolist(), top_logprobs.tolist())
                top_tokens.append(tuple(top_token_info))

            token_logprobs.append(logprobs[token].item())

            from mlx_lm.server import stopping_criteria
            stop_condition = stopping_criteria(
                tokens, stop_id_sequences, self.tokenizer.eos_token_id
            )
            if stop_condition.stop_met:
                finish_reason = "stop"
                if stop_condition.trim_length:
                    stop_sequence_suffix = self.tokenizer.decode(
                        tokens[-stop_condition.trim_length :]
                    )
                    text = text[: -len(stop_sequence_suffix)]
                    full_text = full_text[: -len(stop_sequence_suffix)]
                segment = ""
                break

            if self.stream:
                from mlx_lm.server import sequence_overlap
                if any(
                    (
                        sequence_overlap(tokens, sequence)
                        for sequence in stop_id_sequences
                    )
                ):
                    continue
                elif segment:
                    try:
                        # Simple approach: stop streaming as soon as we see < character in accumulated text
                        if not self.in_tool_call:
                            # Look for any < that could start XML in the accumulated text
                            bracket_pos = text.find("<")
                            if bracket_pos >= 0:
                                self.in_tool_call = True
                                # Calculate what part of this segment comes before the <
                                text_before_segment = text[:-len(segment)] if len(segment) <= len(text) else ""
                                
                                if bracket_pos >= len(text_before_segment):
                                    # The < is in this segment
                                    chars_before_bracket = bracket_pos - len(text_before_segment)
                                    filtered_segment = segment[:chars_before_bracket]
                                else:
                                    # The < was in previous segments, don't send anything
                                    filtered_segment = ""
                            else:
                                # No < found yet, send the segment
                                filtered_segment = segment
                        else:
                            # Already detected <, don't send anything more
                            filtered_segment = ""
                        
                        if filtered_segment:
                            delta_response = {
                                "id": self.request_id,
                                "object": "chat.completion.chunk",
                                "created": self.created,
                                "model": self.requested_model,
                                "system_fingerprint": self.system_fingerprint,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": filtered_segment
                                    },
                                    "finish_reason": None
                                }]
                            }
                            self.wfile.write(f"data: {json.dumps(delta_response)}\n\n".encode())
                            self.wfile.flush()
                        segment = ""
                    except (BrokenPipeError, ConnectionResetError):
                        logging.warning("Client disconnected during streaming")
                        break
                    except Exception as e:
                        logging.error(f"Error sending streaming chunk: {e}")
                        break

        self.prompt_cache.tokens.extend(tokens)

        if gen_response.finish_reason is not None:
            finish_reason = gen_response.finish_reason

        logging.debug(f"Prompt: {gen_response.prompt_tps:.3f} tokens-per-sec")
        logging.debug(f"Generation: {gen_response.generation_tps:.3f} tokens-per-sec")
        logging.debug(f"Peak memory: {gen_response.peak_memory:.3f} GB")
        
        logging.debug(f"FULL GENERATED TEXT: {repr(full_text)}")
        
        # Check if we have XML tool calls in the FULL text
        if "<tool_call>" in full_text or "<function=" in full_text:
            logging.debug("Found XML tool calls in full text, parsing...")
            # Ensure tool parser is initialized
            if self.tool_parser is None:
                self.tool_parser = Qwen3ToolParser(self.tokenizer)
            
            extraction_result = self.tool_parser.extract_tool_calls(full_text, self.body.get("tools", []))
            logging.debug(f"Extraction result: {extraction_result}")
            
            if extraction_result["tools_called"]:
                # Convert to the format expected by the original server
                tool_calls = [json.dumps({"name": tc["function"]["name"], 
                                        "arguments": json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}})
                            for tc in extraction_result["tool_calls"]]
                # Use only the content part as text
                text = extraction_result["content"] or ""
                logging.debug(f"Converted XML to tool_calls: {tool_calls}")
                logging.debug(f"Content text: {repr(text)}")
                
                # Update finish_reason to tool_calls if we have tool calls (like vLLM)
                if len(tool_calls) > 0:
                    finish_reason = "tool_calls"

        if self.stream:
            try:
                final_delta = {"finish_reason": finish_reason}
                
                if tool_calls:
                    # Convert back to OpenAI format for streaming
                    openai_tool_calls = []
                    for i, tool_call_json in enumerate(tool_calls):
                        tc_data = json.loads(tool_call_json)
                        openai_tool_calls.append({
                            "index": i,
                            "type": "function",
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "function": {
                                "name": tc_data["name"],
                                "arguments": json.dumps(tc_data["arguments"])
                            }
                        })
                    final_delta["tool_calls"] = openai_tool_calls
                
                final_response = {
                    "id": self.request_id,
                    "object": "chat.completion.chunk", 
                    "created": self.created,
                    "model": self.requested_model,
                    "system_fingerprint": self.system_fingerprint,
                    "choices": [{
                        "index": 0,
                        "delta": final_delta,
                        "finish_reason": finish_reason
                    }]
                }
                
                self.wfile.write(f"data: {json.dumps(final_response)}\n\n".encode())
                self.wfile.flush()
                
                if self.stream_options is not None and self.stream_options["include_usage"]:
                    usage_response = self.completion_usage_response(len(prompt), len(tokens))
                    self.wfile.write(f"data: {json.dumps(usage_response)}\n\n".encode())
                    self.wfile.flush()
                    
                self.wfile.write("data: [DONE]\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError) as e:
                logging.warning(f"Client disconnected during streaming: {e}")
            except Exception as e:
                logging.error(f"Error during streaming response: {e}")
                try:
                    self.wfile.write("data: [DONE]\n\n".encode())
                    self.wfile.flush()
                except:
                    pass
        else:
            response = self.generate_response(
                text,
                finish_reason,
                len(prompt),
                len(tokens),
                token_logprobs=token_logprobs,
                top_tokens=top_tokens,
                tokens=tokens,
                tool_calls=tool_calls,
            )
            response_json = json.dumps(response).encode()
            indent = "\t"  # Backslashes can't be inside of f-strings
            logging.debug(f"Outgoing Response: {json.dumps(response, indent=indent)}")

            # Send an additional Content-Length header when it is known
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            self.wfile.write(response_json)
            self.wfile.flush()
    def generate_response(
        self,
        text: str,
        finish_reason: Union[str, None],
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
        token_logprobs: Optional[List[float]] = None,
        top_tokens: Optional[List[Dict[int, float]]] = None,
        tokens: Optional[List[int]] = None,
        tool_calls: Optional[List[str]] = None,
    ) -> dict:
        """Enhanced response generation with XML tool parsing."""
        
        logging.debug(f"generate_response called with text: {repr(text[:100])}")
        logging.debug(f"finish_reason: {finish_reason}")
        
        # Initialize tool parser if needed
        if self.tool_parser is None and hasattr(self, 'tokenizer'):
            self.tool_parser = Qwen3ToolParser(self.tokenizer)
        
        # Parse tool calls from the text if any XML format is detected
        parsed_tools = None
        final_content = text
        
        if tool_calls or ("<tool_call>" in text or "<function=" in text):
            logging.debug(f"Detected potential tool calls in text: {repr(text[:200])}")
            # Get tools from the request body for proper parameter validation
            request_tools = self.body.get("tools", [])
            
            # Try to extract XML tool calls from the text
            if self.tool_parser:
                logging.debug("Using existing tool parser")
                extraction_result = self.tool_parser.extract_tool_calls(text, request_tools)
            else:
                logging.warning("Tool parser not initialized, creating temporary one")
                temp_parser = Qwen3ToolParser(self.tokenizer if hasattr(self, 'tokenizer') else None)
                extraction_result = temp_parser.extract_tool_calls(text, request_tools)
            
            logging.debug(f"Extraction result: {extraction_result}")
            
            if extraction_result["tools_called"]:
                parsed_tools = extraction_result["tool_calls"]
                final_content = extraction_result["content"] or ""
                # Set finish reason to tool_calls if we found any (like vLLM)
                finish_reason = "tool_calls"
                logging.debug(f"Found {len(parsed_tools)} tool calls, setting finish_reason to tool_calls")
            else:
                logging.warning("No tools called according to extraction result")
        
        # Use the original MLX server's parse_function logic for any remaining tool_calls
        def parse_function(tool_text):
            tool_call = json.loads(tool_text.strip())
            return {
                "function": {
                    "name": tool_call.get("name", None),
                    "arguments": json.dumps(tool_call.get("arguments", "")),
                },
                "type": "function",
                "id": None,
            }

        # If we still have the old format tool_calls, try to parse them as JSON
        if tool_calls and not parsed_tools:
            try:
                parsed_tools = [parse_function(tool_text) for tool_text in tool_calls]
            except Exception as e:
                logging.warning(f"Failed to parse tool calls: {e}")
                parsed_tools = []
        
        # Generate the base response using parent method logic
        token_logprobs = token_logprobs or []
        top_logprobs = top_tokens or []

        # Static response structure
        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": self.object_type,
            "model": self.requested_model,
            "created": self.created,
            "choices": [
                {
                    "index": 0,
                    "logprobs": {
                        "token_logprobs": token_logprobs,
                        "top_logprobs": top_logprobs,
                        "tokens": tokens,
                    },
                    "finish_reason": finish_reason,
                },
            ],
        }

        if not self.stream:
            if isinstance(prompt_token_count, int) and isinstance(completion_token_count, int):
                response["usage"] = {
                    "prompt_tokens": prompt_token_count,
                    "completion_tokens": completion_token_count,
                    "total_tokens": prompt_token_count + completion_token_count,
                }

        choice = response["choices"][0]

        # Add dynamic response based on completion type
        if self.object_type.startswith("chat.completion"):
            key_name = "delta" if self.stream else "message"
            # Use the original MLX server's exact same logic - always apply parse_function to tool_calls array
            tool_calls_processed = []
            if tool_calls:
                # For the old format tool_calls parameter (list of strings)
                def parse_function_original(tool_text):
                    tool_call = json.loads(tool_text.strip())
                    return {
                        "function": {
                            "name": tool_call.get("name", None),
                            "arguments": json.dumps(tool_call.get("arguments", "")),
                        },
                        "type": "function",
                        "id": None,
                    }
                try:
                    tool_calls_processed = [parse_function_original(tool_text) for tool_text in tool_calls]
                except:
                    tool_calls_processed = []
            elif parsed_tools:
                # For our enhanced XML-parsed tools
                tool_calls_processed = parsed_tools
            else:
                # Empty array like the original
                tool_calls_processed = []
                
            choice[key_name] = {
                "role": "assistant",
                "content": final_content,
                "tool_calls": tool_calls_processed,
            }
        elif self.object_type == "text_completion":
            choice.update(text=final_content)
        else:
            raise ValueError(f"Unsupported response type: {self.object_type}")

        return response


def main():
    parser = argparse.ArgumentParser(description="Enhanced MLX Http Server with Qwen3 Tool Support.")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        help="The path to the MLX model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        help="A model to be used for speculative decoding.",
        default=None,
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Number of tokens to draft when using speculative decoding.",
        default=3,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="",
        help="Specify a chat template for the tokenizer",
        required=False,
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Default sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Default nucleus sampling top-p (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Default top-k sampling (default: 0, disables top-k)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Default min-p sampling (default: 0.0, disables min-p)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Default maximum number of tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Maximum key-value cache size (default: None for unlimited)",
    )
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        help="""A JSON formatted string of arguments for the tokenizer's apply_chat_template, e.g. '{"enable_thinking":false}'""",
        default="{}",
    )
    
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    print(f"Starting enhanced MLX server with Qwen3 tool support...")
    print(f"Model: {args.model}")
    print(f"Host: {args.host}:{args.port}")
    print(f"Tool parsing: XML format -> OpenAI JSON format")
    
    # Use our enhanced handler
    run(args.host, args.port, ModelProvider(args), handler_class=EnhancedAPIHandler)


if __name__ == "__main__":
    main()