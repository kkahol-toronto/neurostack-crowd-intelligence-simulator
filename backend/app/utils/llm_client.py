"""
LLM client wrapper
Unified OpenAI-compatible API calls (OpenAI + Azure OpenAI)
"""

import json
import re
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

from openai import AzureOpenAI, OpenAI

from ..config import Config


def _azure_resource_endpoint(base_url: str) -> str:
    """Turn any Azure OpenAI URL into https://resource.openai.azure.com (no path)."""
    if not base_url:
        return ''
    parsed = urlparse(base_url.strip())
    if not parsed.netloc:
        return base_url.rstrip('/')
    return f"{parsed.scheme}://{parsed.netloc}".rstrip('/')


def _should_use_azure() -> bool:
    if Config.LLM_USE_AZURE:
        return True
    if Config.AZURE_OPENAI_ENDPOINT:
        return True
    bu = (Config.LLM_BASE_URL or '').lower()
    return '.openai.azure.com' in bu


class LLMClient:
    """LLM client"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY is not configured")

        if _should_use_azure():
            endpoint = Config.AZURE_OPENAI_ENDPOINT or _azure_resource_endpoint(self.base_url or '')
            if not endpoint:
                raise ValueError(
                    "Azure OpenAI: set AZURE_OPENAI_ENDPOINT (e.g. https://YOUR_RESOURCE.openai.azure.com) "
                    "or LLM_BASE_URL to a URL containing .openai.azure.com"
                )
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=Config.AZURE_OPENAI_API_VERSION,
                azure_endpoint=endpoint,
            )
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Send a chat completion request

        Args:
            messages: Message list
            temperature: Sampling temperature
            max_tokens: Max tokens
            response_format: Response format (e.g. JSON mode)

        Returns:
            Model text response
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        # Some models (e.g. MiniMax M2.5) embed thinking tags in content; strip them
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Chat request that returns parsed JSON

        Args:
            messages: Message list
            temperature: Sampling temperature
            max_tokens: Max tokens

        Returns:
            Parsed JSON object
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # Strip markdown code fences if present
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON from LLM: {cleaned_response}")
