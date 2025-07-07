import json
import os
from typing import List, Dict

class Tools:
    """Collection of callable tools for agents."""

    @staticmethod
    def get_current_weather(city: str) -> str:
        """Return a placeholder weather report for the given city."""
        return f"It is always sunny in {city}."

    @staticmethod
    def search_files(keyword: str) -> str:
        """Search files in the current directory matching the keyword."""
        matches = []
        for root, _dirs, files in os.walk('.'):
            for fname in files:
                if keyword.lower() in fname.lower():
                    matches.append(os.path.join(root, fname))
        if not matches:
            return f"No files found for '{keyword}'."
        return ", ".join(matches)


def tool_schema() -> List[Dict[str, object]]:
    """Return Ollama tool metadata for all available tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Search local files for a keyword",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "Keyword to search for",
                        }
                    },
                    "required": ["keyword"],
                },
            },
        },
    ]


def tool_descriptions() -> str:
    """Return a human readable list of available tools."""
    return (
        "get_current_weather(city) - Get the current weather for a city.\n"
        "search_files(keyword) - Search local files for a keyword."
    )


def call_tool(name: str, args: Dict[str, object]) -> str:
    """Invoke the named tool with arguments and return its result."""
    if name == "get_current_weather":
        return Tools.get_current_weather(**args)
    if name == "search_files":
        return Tools.search_files(**args)
    raise ValueError(f"Unknown tool {name}")

