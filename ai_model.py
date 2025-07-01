import json
import requests
from typing import List, Dict, Optional

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
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

        parts = [topic_prompt]
        if role_prompt:
            parts.append(role_prompt)
        if chat_style:
            parts.append(f"Use a {chat_style} tone.")
        self.system_prompt = "\n".join(parts)

    def build_prompt(self, chat_log: List[Dict[str, str]]) -> str:
        """Assemble a prompt from system prompt and chat history."""
        lines = [self.system_prompt]
        for entry in chat_log:
            sender = entry.get("sender", "")
            message = entry.get("message", "")
            lines.append(f"{sender}: {message}")
        # Cue for the current AI to speak next
        lines.append(f"{self.name}:")
        # Join with newlines to form the final prompt
        return "\n".join(lines)

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

        try:
            resp = requests.post(
                "http://localhost:11434/api/generate", json=payload, stream=True
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to connect to Ollama: {exc}") from exc

        if resp.status_code != 200:
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
        return result_text
