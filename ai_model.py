import json
import logging
import requests
from typing import List, Dict, Optional

from tools import tool_schema, tool_descriptions, call_tool

from runtime_utils import create_object_logger


def model_supports_tools(model_id: str) -> bool:
    """Return True if the model is known to support tool calling."""
    id_lower = model_id.lower()
    supported_keywords = ["nemo", "llama3", "tool"]
    return any(k in id_lower for k in supported_keywords)

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
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.watchdog_timeout = watchdog_timeout

        self.supports_tools = model_supports_tools(model_id)

        self.logger = create_object_logger(self.__class__.__name__)
        self.logger.info(
            "Initialized AIModel for %s using %s", self.name, self.model_id
        )

        parts = [topic_prompt]
        if role_prompt:
            parts.append(role_prompt)
        if self.supports_tools:
            parts.append("You have access to the following tools:\n" + tool_descriptions())
        if chat_style:
            parts.append(f"Use a {chat_style} tone.")
        self.system_prompt = "\n".join(parts)

    def build_prompt(self, chat_log: List[Dict[str, str]]) -> str:
        """Assemble a prompt from system prompt and chat history."""
        lines = [self.system_prompt]
        for entry in chat_log:
            sender = entry.get("sender", "")
            message = entry.get("message", "")
            lines.append(f"{message}")
        # Cue for the current AI to speak next
        lines.append(f"{self.name}:")
        # Join with newlines to form the final prompt
        prompt = "\n".join(lines)
        self.logger.debug("Built prompt of %d characters", len(prompt))
        return prompt

    def generate_response(self, chat_log: List[Dict[str, str]]) -> str:
        """Generate a response from the model using Ollama's API."""
        prompt = self.build_prompt(chat_log)

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "system": self.system_prompt,
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        self.logger.debug("Sending generation request")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Payload to Ollama:\n%s", json.dumps(payload, indent=2))
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                stream=True,
                timeout=self.watchdog_timeout,
            )
        except requests.RequestException as exc:
            self.logger.error("Connection error: %s", exc)
            raise RuntimeError(f"Failed to connect to Ollama: {exc}") from exc

        if resp.status_code != 200:
            self.logger.error(
                "Ollama API error: %s %s", resp.status_code, resp.text
            )
            raise RuntimeError(
                f"Ollama API error: {resp.status_code} {resp.text}"
            )

        result_text = ""
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if chunk.get("done"):
                break
            result_text += chunk.get("response", "")
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
            payload["tools"] = tools
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Payload to Ollama:\n%s", json.dumps(payload, indent=2))
        try:
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json=payload,
                timeout=self.watchdog_timeout,
            )
        except requests.RequestException as exc:
            self.logger.error("Connection error: %s", exc)
            raise RuntimeError(f"Failed to connect to Ollama: {exc}") from exc

        if resp.status_code != 200:
            self.logger.error(
                "Ollama API error: %s %s", resp.status_code, resp.text
            )
            raise RuntimeError(
                f"Ollama API error: {resp.status_code} {resp.text}"
            )

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:  # noqa: BLE001
            self.logger.error("Invalid JSON from Ollama: %s", exc)
            raise RuntimeError("Invalid JSON from Ollama") from exc

        return data


class Agent:
    """Base class for all agents."""

    def __init__(
        self,
        name: str,
        model_name: str,
        role_prompt: str,
        config: Dict[str, Optional[str]],
    ) -> None:
        self.name = name
        self.model_name = model_name
        self.role_prompt = role_prompt
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
        if not self.model.supports_tools:
            self.logger.info("Model lacks tool support, falling back to text")
            return self.model.generate_response(context)

        messages: List[Dict[str, object]] = [
            {"role": "system", "content": self.model.system_prompt}
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
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {}
                result = call_tool(str(name), args)
                messages.append({"role": "tool", "name": name, "content": result})


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

