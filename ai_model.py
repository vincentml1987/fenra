import json
import logging
import requests
from typing import List, Dict, Optional

from tools import tool_schema, tool_descriptions, call_tool

from runtime_utils import (
    create_object_logger,
    generate_with_watchdog,
    parse_model_size,
    strip_think_markup,
    WATCHDOG_TRACKER,
)


class AIModel:
    """A single AI agent powered by an Ollama model."""

    def __init__(
        self,
        name: str,
        model_id: str,
        topic_prompt: str,
        role_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 300,
        chat_style: Optional[str] = None,
        watchdog_timeout: int = 300,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.watchdog_timeout = watchdog_timeout

        self.logger = create_object_logger(self.__class__.__name__)
        self.logger.info(
            "Initialized AIModel for %s using %s", self.name, self.model_id
        )

        parts = [topic_prompt]
        if chat_style:
            parts.append(f"Use a {chat_style} tone.")
        if role_prompt:
            parts.append(role_prompt)
        self.base_prompt = "\n".join(parts)

        # system_prompt is optional and comes from config
        self.system_prompt = system_prompt

    def build_prompt(self, chat_log: List[Dict[str, str]]) -> str:
        """Assemble a prompt from system prompt and chat history."""
        lines = ["=====Chat Begins====="]
        for entry in chat_log:
            sender = entry.get("sender", "")
            message = entry.get("message", "")
            timestamp = entry.get("timestamp", "")
            lines.append(f"[{timestamp}] {sender}: {message}")
            lines.append("-" * 80)
        lines.append("=====Chat Ends=====")
        lines.append(
            "The above message is the full chat log. Each message is separated by a series of hyphens. The names of the speakers are indicated before the message."
        )
        lines.append(self.base_prompt)
        lines.append(f"Your name is {self.name}.")

        prompt = "\n".join(lines)
        self.logger.debug("Built prompt of %d characters", len(prompt))
        return prompt

    def generate_response(self, chat_log: List[Dict[str, str]]) -> str:
        """Generate a response from the model using Ollama's API."""
        prompt = self.build_prompt(chat_log)

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.system_prompt:
            payload["system"] = self.system_prompt
        self.logger.debug("Sending generation request")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Payload to Ollama:\n%s", json.dumps(payload, indent=2))
        result_text = generate_with_watchdog(
            payload,
            parse_model_size(self.model_id),
            WATCHDOG_TRACKER,
        )
        result_text = strip_think_markup(result_text)
        self.logger.debug("Generated %d characters", len(result_text))
        return result_text

    def chat_completion(
        self,
        messages: List[Dict[str, object]],
        tools: Optional[List[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        """Call Ollama chat API optionally with tools."""
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if tools:
            parts.append("You have access to the following tools:\n" + tool_descriptions())
            payload["tools"] = tools
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Payload to Ollama:\n%s", json.dumps(payload, indent=2))
        result_text = generate_with_watchdog(
            payload,
            parse_model_size(self.model_id),
            WATCHDOG_TRACKER,
        )
        try:
            data = json.loads(result_text)
        except json.JSONDecodeError as exc:  # noqa: BLE001
            self.logger.error("Invalid JSON from Ollama: %s", exc)
            raise RuntimeError("Invalid JSON from Ollama") from exc
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = strip_think_markup(content)
        return data


class Agent:
    """Base class for all agents."""

    def __init__(
        self,
        name: str,
        model_name: str,
        role_prompt: str,
        config: Dict[str, Optional[str]],
        groups: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.model_name = model_name
        self.role_prompt = role_prompt
        self.groups = groups or ["general"]
        self.active = True
        self.logger = create_object_logger(self.__class__.__name__)
        self.logger.info("Initialized agent %s", self.name)

        self.model = AIModel(
            name=name,
            model_id=model_name,
            topic_prompt=config.get("topic_prompt", ""),
            role_prompt=role_prompt,
            temperature=float(config.get("temperature", 0.7)),
            max_tokens=int(config.get("max_tokens", 300)),
            chat_style=config.get("chat_style"),
            watchdog_timeout=int(config.get("watchdog_timeout", 300)),
            system_prompt=config.get("system_prompt"),
        )

    def step(self, context: List[Dict[str, str]]):
        raise NotImplementedError


class Ruminator(Agent):
    """Regular discussion participant."""

    def step(self, context: List[Dict[str, str]]) -> str:
        self.logger.info("Generating response")
        reply = self.model.generate_response(context)
        self.logger.debug("Response length %d", len(reply))
        return reply


class ToolAgent(Agent):
    """Agent capable of using tools via the Ollama API."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tool_schema()

    def step(self, context: List[Dict[str, str]]) -> str:
        messages: List[Dict[str, object]] = [
            {"role": "system", "content": self.model.base_prompt}
        ]
        for entry in context:
            sender = entry.get("sender", "")
            content = entry.get("message", "")
            role = "assistant" if sender == self.name else "user"
            messages.append({"role": role, "content": content})

        while True:
            resp = self.model.chat_completion(messages, tools=self.tools)
            msg = resp.get("message", {})
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                messages.append({"role": "assistant", "content": content})
                return content

            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name")
                call_id = call.get("id")
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {}
                result = call_tool(str(name), args)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": result})


class Archivist(Agent):
    """Non-participating summarizer and archivist."""

    def step(self, full_context: List[Dict[str, str]]) -> str:
        """Archive transcript and return compressed summary."""
        from datetime import datetime
        import os

        self.logger.info("Archiving full transcript")

        # Archive full transcript
        try:
            os.makedirs("archive", exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fname = os.path.join("archive", f"{ts}_full.txt")
            with open(fname, "w", encoding="utf-8") as f:
                for entry in full_context:
                    sender = entry.get("sender", "")
                    message = entry.get("message", "")
                    timestamp = entry.get("timestamp", "")
                    f.write(f"[{timestamp}] {sender}: {message}\n")
        except OSError as exc:
            self.logger.error("Failed to archive conversation: %s", exc)

        # Generate summary using the model
        summary = self.model.generate_response(full_context)

        # Save summary to file
        try:
            with open("summary.txt", "w", encoding="utf-8") as f:
                f.write(summary)
        except OSError as exc:
            self.logger.error("Failed to write summary: %s", exc)

        # Append summary with timestamp to running log
        try:
            with open("summary_log.txt", "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {summary}\n{'-' * 80}\n")
        except OSError as exc:
            self.logger.error("Failed to update summary log: %s", exc)

        self.logger.info("Summary generated")
        return summary

