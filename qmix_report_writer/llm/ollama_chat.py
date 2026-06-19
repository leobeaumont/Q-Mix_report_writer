import logging

import aiohttp
from typing import List, Union, Optional, Dict

from .format import Message
from .llm import LLM
from qmix_report_writer.utils.config import get_llm_config, get_config
from qmix_report_writer.utils.globals import PromptTokens, CompletionTokens

logger = logging.getLogger("llm.ollama_chat")


def _trim_truncated_tail(text: str) -> str:
    """Cut a max_tokens-truncated completion back to its last complete sentence.

    A response cut mid-sentence (e.g. a review ending in "(5)") confuses every
    downstream consumer. Trimming to the last sentence boundary keeps the output
    coherent. If no boundary exists in the second half of the text, it is
    returned unchanged — losing half the content is worse than a ragged tail.
    """
    cut = max(text.rfind(ch) for ch in ".!?\n")
    if cut >= len(text) // 2:
        return text[: cut + 1].rstrip()
    return text


def _build_ollama_endpoint(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def _get_ollama_endpoint(calling_agent: Optional[str] = None) -> str:
    """Read the Ollama base URL from default.yaml at call time."""
    if calling_agent:
        config = get_config()
        agent_config = config.get("agent_configs", {}).get("redacting", {})
        respective_urls = agent_config.get("respective_ollama_urls", {})
        base_url = respective_urls.get(calling_agent, None)
        if base_url:
            return _build_ollama_endpoint(base_url)
    llm_config = get_llm_config()
    base_url = llm_config.get("providers", {}).get("ollama", {}).get("base_url", "http://localhost:11434")
    return _build_ollama_endpoint(base_url)


def _get_agent_max_tokens(calling_agent: Optional[str], default: int) -> int:
    """Return per-agent max_tokens from respective_max_tokens config, or the default."""
    if calling_agent:
        config = get_config()
        agent_config = config.get("agent_configs", {}).get("redacting", {})
        per_agent = agent_config.get("respective_max_tokens", {})
        if calling_agent in per_agent:
            return int(per_agent[calling_agent])
    return default


async def achat_ollama(
    model_name: str,
    messages: list,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    response_schema: Optional[Dict] = None,
    calling_agent: Optional[str] = None,
) -> str:
    """Send a query to a LLM using oLLama."""
    endpoint = _get_ollama_endpoint(calling_agent)
    max_tokens = _get_agent_max_tokens(calling_agent, max_tokens)
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "model": model_name,
        "messages": [m.to_dict() if isinstance(m, Message) else m for m in messages],
        "stream": False,
        "max_tokens": max_tokens,
        "options": {
            "frequency_penalty": 0.6,  # Discourage repeating the exact same phrases
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
        }
    }

    if response_schema:
        data["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "scores",
                "strict": True,
                "schema": response_schema
            }
        }

    custom_timeout = aiohttp.ClientTimeout(total=600, connect=60, sock_read=600)

    async with aiohttp.ClientSession(timeout=custom_timeout) as session:
        async with session.post(endpoint, headers=headers, json=data) as response:
            if "application/json" not in (response.headers.get("Content-Type") or ""):
                text = await response.text()
                raise aiohttp.ContentTypeError(
                    response.request_info,
                    history=response.history,
                    message=f"Unexpected content-type from Ollama at {endpoint}. Body: {text[:200]}",
                )
            response_data = await response.json()
            if "choices" not in response_data:
                error_message = response_data.get("error", {}).get("message", "Unknown error from Ollama")
                raise Exception(f"Ollama API Error: {error_message}")

            # Track token usage from Ollama's response into the global singletons
            # so graph.arun() can compute total tokens and QMIX observations stay accurate
            usage = response_data.get("usage", {})
            PromptTokens.instance().value += usage.get("prompt_tokens", 0)
            CompletionTokens.instance().value += usage.get("completion_tokens", 0)

            choice = response_data["choices"][0]
            msg = choice["message"]
            completion = msg.get("content") or ""
            if not completion.strip():
                finish_reason = choice.get("finish_reason", "unknown")
                completion_tokens = usage.get("completion_tokens", "unknown")
                print(f"[DEBUG empty response] finish_reason={finish_reason}, "
                      f"completion_tokens={completion_tokens}, msg_keys={list(msg.keys())}, "
                      f"msg={msg}")
                raise ValueError(
                    f"Empty response from model "
                    f"(finish_reason={finish_reason}, "
                    f"completion_tokens={completion_tokens})"
                )

            # Detect a hard cut at the max_tokens ceiling. The response is kept
            # (a retry would hit the same cap) but trimmed to the last complete
            # sentence so downstream agents never consume a mid-sentence tail.
            # JSON-schema responses are left untouched — trimming breaks JSON.
            if choice.get("finish_reason") == "length" and not response_schema:
                logger.warning(
                    f"Response truncated at max_tokens={max_tokens} "
                    f"(agent={calling_agent or 'unknown'}, "
                    f"completion_tokens={usage.get('completion_tokens', '?')}) — "
                    f"trimming to the last complete sentence."
                )
                completion = _trim_truncated_tail(completion)
            return completion


class OllamaChat(LLM):
    def __init__(self, model_name: str, temperature: float = 0.7, top_p: float = 0.8,
                 top_k: int = 20, min_p: float = 0.0, max_tokens: int = 4096,
                 response_schema: Optional[Dict] = None):
        self.model_name = model_name
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._min_p = min_p
        self._max_tokens = max_tokens
        self._response_schema = response_schema

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_schema: Optional[Dict] = None,
        num_comps: Optional[int] = None,
        calling_agent: Optional[str] = None,
    ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self._max_tokens
        if temperature is None:
            temperature = self._temperature
        if response_schema is None:
            response_schema = self._response_schema

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        elif isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
            messages = [Message(role=m["role"], content=m["content"]) for m in messages]

        return await achat_ollama(
            self.model_name,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self._top_p,
            top_k=self._top_k,
            min_p=self._min_p,
            response_schema=response_schema,
            calling_agent=calling_agent
        )

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_schema: Optional[Dict] = None,
        num_comps: Optional[int] = None,
        calling_agent: Optional[str] = None,
    ) -> Union[List[str], str]:
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self.agen(messages, max_tokens, temperature, response_schema, num_comps, calling_agent)).result()
        return asyncio.run(self.agen(messages, max_tokens, temperature, response_schema, num_comps, calling_agent))


if __name__ == "__main__":
    print(_get_ollama_endpoint("LeadArchitect"))
    print(_get_ollama_endpoint("Researcher"))
