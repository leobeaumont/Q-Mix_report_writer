import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt

from .format import Message
from .llm import LLM
from .llm_registry import LLMRegistry
from utils.config import get_llm_config
from utils.globals import PromptTokens, CompletionTokens


def _build_ollama_endpoint(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def _get_ollama_endpoint() -> str:
    """Read the Ollama base URL from default.yaml at call time."""
    llm_config = get_llm_config()
    base_url = llm_config.get("ollama_base_url", "http://localhost:11434")
    return _build_ollama_endpoint(base_url)


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=60))
async def achat_ollama(model_name: str, messages: list, max_tokens: int = 1024, temperature: float = 0.2, json_mode: bool = False):
    """Send a query to a LLM using oLLama."""
    endpoint = _get_ollama_endpoint()
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "model": model_name,
        "messages": [m.to_dict() if isinstance(m, Message) else m for m in messages],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if json_mode:
        data["response_format"] = {"type": "json_object"}

    async with aiohttp.ClientSession() as session:
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

            msg = response_data["choices"][0]["message"]
            completion = msg.get("content") or ""
            return completion


# Register your locally available Ollama models here.
# The name must match exactly what you have pulled in Ollama (run `ollama list` to check).
@LLMRegistry.register("tinyllama")
@LLMRegistry.register("llama3.2")
@LLMRegistry.register("llama3.1")
@LLMRegistry.register("mistral")
@LLMRegistry.register("qwen2.5")
@LLMRegistry.register("deepseek-r1")
@LLMRegistry.register("phi4")
class OllamaChat(LLM):
    def __init__(self, model_name: str, temperature: float = 0.2, max_tokens: int = 1024, json_mode: bool = False):
        self.model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._json_mode = json_mode

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        json_mode: Optional[bool] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self._max_tokens
        if temperature is None:
            temperature = self._temperature
        if json_mode is None: 
            json_mode = self._json_mode

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        elif isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
            messages = [Message(role=m["role"], content=m["content"]) for m in messages]

        return await achat_ollama(self.model_name, messages, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode)

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        json_mode: Optional[bool] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self.agen(messages, max_tokens, temperature, json_mode, num_comps)).result()
        return asyncio.run(self.agen(messages, max_tokens, temperature, json_mode, num_comps))
