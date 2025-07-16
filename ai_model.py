import json
import logging
import re
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

logger = logging.getLogger(__name__)


class AIModel:
    """A single AI agent powered by an Ollama model."""

    def __init__(
        self,
        name: str,
        model_id: str,
        topic_prompt: str,
        role_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        chat_style: Optional[str] = None,
        watchdog_timeout: int = 300,
        system_prompt: Optional[str] = None,
    ) -> None:
        logger.debug(
            "Entering AIModel.__init__ with name=%s model_id=%s topic_prompt=%s role_prompt=%s temperature=%s max_tokens=%s chat_style=%s watchdog_timeout=%s system_prompt=%s",
            name,
            model_id,
            topic_prompt,
            role_prompt,
            temperature,
            max_tokens,
            chat_style,
            watchdog_timeout,
            system_prompt,
        )
        self.name = name
        self.model_id = model_id
        self.model_size = parse_model_size(model_id)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.watchdog_timeout = watchdog_timeout

        self.logger = create_object_logger(self.__class__.__name__)
        self.logger.info(
            "Initialized AIModel for %s using %s", self.name, self.model_id
        )

        self.role_prompt = role_prompt
        self.topic_prompt = topic_prompt

        parts = []
        if chat_style:
            parts.append(f"Use a {chat_style} tone.")
        self.base_prompt = "\n".join(parts)

        # system_prompt is optional and comes from config
        self.system_prompt = system_prompt

        self.logger.debug("Exiting AIModel.__init__")

    def build_prompt(self, chat_log: List[Dict[str, str]]) -> str:
        """Assemble a prompt from system prompt and chat history."""
        self.logger.debug("Entering build_prompt with chat_log=%s", chat_log)
        lines = ["=====Chat Begins====="]
        for entry in chat_log:
            message = entry.get("message", "")
            lines.append(message)
            lines.append("-" * 80)
        lines.append("=====Chat Ends=====")
        lines.append(
            "The above message is the full chat log. Each message is separated by a series of hyphens."
        )
        if self.base_prompt:
            lines.append(self.base_prompt)
        if self.max_tokens is not None:
            lines.append(
                f"Keep your response to less than {self.max_tokens} words. Otherwise, your response will be truncated."
            )

        prompt = "\n".join(lines)
        self.logger.debug("Built prompt of %d characters", len(prompt))
        self.logger.debug("Exiting build_prompt")
        return prompt

    def generate_response(self, chat_log: List[Dict[str, str]]) -> str:
        """Generate a response from the model using Ollama's API."""
        self.logger.debug("Entering generate_response with chat_log=%s", chat_log)
        prompt = self.build_prompt(chat_log)

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature,
            "options": {},
        }
        system_parts = []
        if self.system_prompt:
            system_parts.append(self.system_prompt)
        role_topic = " ".join(
            [p for p in [self.role_prompt, self.topic_prompt] if p]
        )
        if role_topic:
            system_parts.append(role_topic)
        if system_parts:
            payload["system"] = "\n".join(system_parts)
        self.logger.debug("Sending generation request")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Payload to Ollama:\n%s", json.dumps(payload, indent=2))
        result_text = generate_with_watchdog(
            payload,
            self.model_size,
            WATCHDOG_TRACKER,
        )
        result_text = strip_think_markup(result_text)
        self.logger.debug("Generated %d characters", len(result_text))
        self.logger.debug("Exiting generate_response")
        return result_text

    def generate_from_prompt(
        self,
        prompt: str,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        *,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> str:
        """Generate a response from a custom prompt."""
        self.logger.debug(
            "Entering generate_from_prompt with num_ctx=%s num_predict=%s",
            num_ctx,
            num_predict,
        )
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature if temperature is None else temperature,
            "options": {},
        }
        if num_ctx is not None and self.max_tokens is not None:
            payload["options"]["num_ctx"] = num_ctx
        # num_predict is accepted for compatibility but intentionally ignored
        if system is None:
            system_parts = []
            if self.system_prompt:
                system_parts.append(self.system_prompt)
            role_topic = " ".join(
                [p for p in [self.role_prompt, self.topic_prompt] if p]
            )
            if role_topic:
                system_parts.append(role_topic)
            if system_parts:
                payload["system"] = "\n".join(system_parts)
        elif system:
            payload["system"] = system
        # if system is empty string, omit the system field entirely
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Payload to Ollama:\n%s", json.dumps(payload, indent=2))
        result_text = generate_with_watchdog(
            payload,
            self.model_size,
            WATCHDOG_TRACKER,
        )
        result_text = strip_think_markup(result_text)
        self.logger.debug("Generated %d characters", len(result_text))
        self.logger.debug("Exiting generate_from_prompt")
        return result_text

    def chat_completion(
        self,
        messages: List[Dict[str, object]],
        tools: Optional[List[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        """Call Ollama chat API optionally with tools."""
        self.logger.debug(
            "Entering chat_completion with messages=%s tools=%s",
            messages,
            tools,
        )
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "options": {},
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        system_parts = []
        if self.system_prompt:
            system_parts.append(self.system_prompt)
        role_topic = " ".join(
            [p for p in [self.role_prompt, self.topic_prompt] if p]
        )
        if role_topic:
            system_parts.append(role_topic)
        if system_parts:
            payload["system"] = "\n".join(system_parts)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Payload to Ollama:\n%s", json.dumps(payload, indent=2))
        result_text = generate_with_watchdog(
            payload,
            self.model_size,
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
        self.logger.debug("Exiting chat_completion")
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
        logger.debug(
            "Entering Agent.__init__ with name=%s model_name=%s role_prompt=%s groups=%s",
            name,
            model_name,
            role_prompt,
            groups,
        )
        self.name = name
        self.model_name = model_name
        self.role_prompt = role_prompt
        self.groups = groups or ["general"]
        self.active = True
        self.logger = create_object_logger(self.__class__.__name__)
        self.logger.info("Initialized agent %s", self.name)

        max_tok_val = config.get("max_tokens")
        max_tok = int(max_tok_val) if max_tok_val is not None else None
        self.model = AIModel(
            name=name,
            model_id=model_name,
            topic_prompt=config.get("topic_prompt", ""),
            role_prompt=role_prompt,
            temperature=float(config.get("temperature", 0.7)),
            max_tokens=max_tok,
            chat_style=config.get("chat_style"),
            watchdog_timeout=int(config.get("watchdog_timeout", 300)),
            system_prompt=config.get("system_prompt"),
        )

        self.logger.debug("Exiting Agent.__init__")

    def step(self, context: List[Dict[str, str]]):
        self.logger.debug("Entering Agent.step with context=%s", context)
        self.logger.debug("Exiting Agent.step")
        raise NotImplementedError


class Ruminator(Agent):
    """Regular discussion participant."""

    def step(self, context: List[Dict[str, str]]) -> str:
        self.logger.debug("Entering Ruminator.step with context=%s", context)
        self.logger.info("Generating response")
        reply = self.model.generate_response(context)
        self.logger.debug("Response length %d", len(reply))
        self.logger.debug("Exiting Ruminator.step")
        return reply


class ToolAgent(Agent):
    """Agent capable of using tools via the Ollama API."""

    def __init__(self, *args, **kwargs) -> None:
        logger.debug("Entering ToolAgent.__init__ with args=%s kwargs=%s", args, kwargs)
        super().__init__(*args, **kwargs)
        self.tools = tool_schema()
        self.logger.debug("Exiting ToolAgent.__init__")

    def step(self, context: List[Dict[str, str]]) -> str:
        self.logger.debug("Entering ToolAgent.step with context=%s", context)
        parts = []
        role_topic = " ".join(
            [p for p in [self.model.role_prompt, self.model.topic_prompt] if p]
        )
        if role_topic:
            parts.append(role_topic)
        if self.model.base_prompt:
            parts.append(self.model.base_prompt)
        if self.model.max_tokens is not None:
            parts.append(
                f"Keep your response to less than {self.model.max_tokens} words. Otherwise, your response will be truncated."
            )
        system_msg = "\n".join(parts)
        messages: List[Dict[str, object]] = [
            {"role": "system", "content": system_msg}
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
        self.logger.debug("Exiting ToolAgent.step")


class Archivist(Agent):
    """Non-participating summarizer and archivist."""

    def step(self, full_context: List[Dict[str, str]]) -> str:
        """Archive transcript and return compressed summary."""
        self.logger.debug(
            "Entering Archivist.step with full_context=%s", full_context
        )
        from datetime import datetime
        import os

        self.logger.info("Archiving full transcript")

        lines = []
        prompt_lines = []
        for entry in full_context:
            sender = entry.get("sender", "")
            message = entry.get("message", "")
            timestamp = entry.get("timestamp", "")
            lines.append(f"[{timestamp}] {sender}: {message}")
            prompt_lines.append(message)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        try:
            os.makedirs("archive", exist_ok=True)
            fname = os.path.join("archive", f"{ts}_full.txt")
            with open(fname, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
        except OSError as exc:
            self.logger.error("Failed to archive conversation: %s", exc)

        prompt_lines.append(
            "Your job is to summarize the above chat. Keep as much of the meaning of the conversation as you can. You are not responding to anyone, so do not actually speak directly to anyone. Simply summarize the chat."
        )
        prompt = "\n".join(prompt_lines)
        word_count = len(prompt.split())

        summary = self.model.generate_from_prompt(prompt, num_ctx=word_count)

        try:
            with open("summary.txt", "w", encoding="utf-8") as f:
                f.write(summary)
        except OSError as exc:
            self.logger.error("Failed to write summary: %s", exc)

        try:
            with open("summary_log.txt", "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {summary}\n{'-' * 80}\n")
        except OSError as exc:
            self.logger.error("Failed to update summary log: %s", exc)

        self.logger.info("Summary generated")
        self.logger.debug("Exiting Archivist.step")
        return summary


class Listener(Agent):
    """Agent that monitors user questions and notifies other AIs."""

    CHECK_INSTRUCTIONS = (
        "You are a Listener AI. You are not speaking to a human. "
        "Determine if the user's question has been answered in the output to the world. "
        "The user's message displays under -----Message from User----- "
        "Messages sent to the user appear under -----Output to World----- "
        "Reply with 'Yes' if the output contains the answer to he message from the user. Reply with 'No' if it has not. "
        "Do not respond with anything but 'Yes' or 'No'."
    )

    PROMPT_INSTRUCTIONS = (
        "You are a Listener AI speaking to other AIs. "
        "The user's message displays under -----Message from User----- "
        "Gently remind the other AIs that the user asked a question and restate the question in your own words."
    )

    CLEAR_INSTRUCTIONS = (
        "You are a Listener AI speaking to other AIs. "
        "The user's message displays under -----Message from User----- "
        "Let the other AIs know that the users request has been addressed."
    )

    def check_answered(self, message: str, outputs: List[str]) -> bool:
        """Return True if the user's question appears answered."""
        if not outputs:
            return False
        for out in outputs:
            print(out)
            lines = [
                (
                    "Below is a message received from the humans and a message "
                    "sent to the humans. Does the sent message respond to the "
                    "received message? Only answer yes or no:"
                ),
                "-----------------",
                f"Received Message: {message}",
                "-----------------",
                f"Sent Message: {out}",
            ]
            prompt = "\n".join(lines)
            wc = len(prompt.split())
            reply = self.model.generate_from_prompt(
                prompt,
                num_ctx=wc,
                num_predict=3,
                temperature=0.0,
                system="",
            )

            logger.debug("Listener responded: %s", reply)
            cleaned = re.sub(r"[^a-zA-Z]", "", reply).lower()
            if "yes" in cleaned:
                return True
        return False

    def prompt_ais(self, transcript: str, message: str) -> str:
        """Ask other AIs to answer the user's question."""
        lines = ["-----Message from User-----"]
        lines.append(message)
        lines.append("-----Your Instructions-----")
        lines.append(self.PROMPT_INSTRUCTIONS)
        prompt = "\n".join(lines)
        reply = self.model.generate_from_prompt(
            prompt,
            system="",
        )
        return reply

    def clear_ais(self, message: str) -> str:
        """Notify other AIs that the user's request has been addressed."""
        lines = ["-----Message from User-----"]
        lines.append(message)
        lines.append("-----Your Instructions-----")
        lines.append(self.CLEAR_INSTRUCTIONS)
        prompt = "\n".join(lines)
        reply = self.model.generate_from_prompt(prompt, system="")
        return reply

