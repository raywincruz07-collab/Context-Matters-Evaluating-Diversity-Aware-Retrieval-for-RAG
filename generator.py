"""
LLM generator for an OpenAI-compatible LiteLLM endpoint.

This replaces Groq. It calls the university GPU-backed endpoint used in the
provided R code:
    https://maki.uni-mannheim.de/v1/chat/completions

Do NOT hardcode API keys in this file. Use the Streamlit sidebar or set:
    $env:MAKI_API_KEY = "your_key_here"        # Windows PowerShell
    export MAKI_API_KEY="your_key_here"        # Linux/Mac/Git Bash
"""

from __future__ import annotations

import json
import time
from typing import Dict, Iterator, List, Optional

import requests

from config import MAKI_API_KEY, MAKI_HOST, MAKI_MODEL, MAKI_DEFAULT_CTX, PROMPT_TEMPLATE


class MakiGenerator:
    """Generator using the OpenAI-compatible /chat/completions API."""

    def __init__(
        self,
        model: str = MAKI_MODEL,
        api_key: str = MAKI_API_KEY,
        host: str = MAKI_HOST,
        default_ctx: int = MAKI_DEFAULT_CTX,
        timeout: int = 300,
    ):
        self.model = model
        self.api_key = api_key
        self.host = host.rstrip("/")
        self.default_ctx = default_ctx
        self.timeout = timeout

    @property
    def chat_url(self) -> str:
        return f"{self.host}/chat/completions"

    def is_available(self) -> bool:
        """Check whether an API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    def list_models(self) -> List[str]:
        """Optional model listing. Some LiteLLM deployments disable this."""
        if not self.is_available():
            return []
        try:
            r = requests.get(
                f"{self.host}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            return [m.get("id", "") for m in data.get("data", []) if m.get("id")]
        except Exception:
            return []

    def _build_prompt(self, query: str, context_docs: List[Dict]) -> str:
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            text = doc.get("text", str(doc)) if isinstance(doc, dict) else str(doc)
            meta = ""
            if isinstance(doc, dict):
                section = doc.get("section")
                pubid = doc.get("pubid") or doc.get("pmid")
                if section or pubid:
                    meta = f" ({section or 'section'}; PMID/PubMed: {pubid or 'unknown'})"
            context_parts.append(f"[Document {i}{meta}]:\n{text}")
        context = "\n\n".join(context_parts)
        return PROMPT_TEMPLATE.format(context=context, question=query)

    def _headers(self) -> Dict[str, str]:
        if not self.is_available():
            raise ValueError("MAKI_API_KEY is not set.")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self,
        query: str,
        context_docs: List[Dict],
        temperature: float,
        max_tokens: int,
        stream: bool = False,
    ) -> Dict:
        prompt = self._build_prompt(query, context_docs)
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            # This mirrors the R code's num_ctx = default_ctx behavior.
            # LiteLLM/model servers that do not use this field usually ignore it.
            "num_ctx": self.default_ctx,
            "stream": stream,
        }

    def generate(
        self,
        query: str,
        context_docs: List[Dict],
        temperature: float = 0.1,
        max_tokens: int = 512,
        max_retries: int = 3,
        retry_sleep: int = 5,
    ) -> str:
        """Generate one answer using the GPU-backed API endpoint."""
        if not self.is_available():
            return "Error: MAKI_API_KEY not set. Enter it in the sidebar or set $env:MAKI_API_KEY."

        payload = self._payload(query, context_docs, temperature, max_tokens, stream=False)

        last_error: Optional[str] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    self.chat_url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            except requests.HTTPError as e:
                body = ""
                try:
                    body = response.text[:500]
                except Exception:
                    pass
                last_error = f"HTTP {getattr(response, 'status_code', 'unknown')}: {body or str(e)}"
            except Exception as e:
                last_error = str(e)

            if attempt < max_retries:
                time.sleep(retry_sleep)

        return f"Error: University GPU API call failed after {max_retries} attempts. {last_error}"

    def generate_streaming(
        self,
        query: str,
        context_docs: List[Dict],
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> Iterator[str]:
        """Generate answer in streaming mode. Falls back to normal call if streaming fails."""
        if not self.is_available():
            yield "Error: MAKI_API_KEY not set. Enter it in the sidebar or set $env:MAKI_API_KEY."
            return

        payload = self._payload(query, context_docs, temperature, max_tokens, stream=True)

        try:
            with requests.post(
                self.chat_url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
                stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line = line[len("data: "):]
                    if line.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                        token = chunk["choices"][0].get("delta", {}).get("content")
                        if token:
                            yield token
                    except Exception:
                        continue
        except Exception:
            # Some LiteLLM endpoints do not support streaming. Do not crash the UI.
            yield self.generate(query, context_docs, temperature=temperature, max_tokens=max_tokens)


# Backward-compatible alias so old imports still work.
GroqGenerator = MakiGenerator
