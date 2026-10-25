import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Optional
import httpx

from src.config import get_settings
from src.core.exceptions import LLMGenerationError

logger = logging.getLogger(__name__)
settings = get_settings()

SYSTEM_RAG_PROMPT = """You are an accurate, helpful AI assistant.
Answer the user's question strictly based on the provided context below.
If the answer cannot be determined from the context, state honestly that you do not have enough information.
Always cite the relevant source/chunk if applicable.

Context:
{context}

Question:
{question}
"""


class BaseLLMService(ABC):
    """Abstract base class for LLM generation."""

    @abstractmethod
    async def generate(self, question: str, context: str) -> str:
        """Returns the full text completion."""
        pass

    @abstractmethod
    async def generate_stream(
        self, question: str, context: str
    ) -> AsyncGenerator[str, None]:
        """Asynchronously streams generated tokens."""
        pass


class GeminiLLMService(BaseLLMService):
    """LLM Service powered by Google Gemini REST API."""

    def __init__(self, api_key: str, model_name: str = "gemini-3-flash-preview"):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    async def generate(self, question: str, context: str) -> str:
        prompt = SYSTEM_RAG_PROMPT.format(context=context, question=question)
        url = f"{self.base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, timeout=30.0)
                if resp.status_code != 200:
                    raise LLMGenerationError(
                        f"Gemini API returned {resp.status_code}: {resp.text}"
                    )
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return text
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise LLMGenerationError(f"Gemini generation error: {e}") from e

    async def generate_stream(
        self, question: str, context: str
    ) -> AsyncGenerator[str, None]:
        # Generate completion and stream token by token
        full_text = await self.generate(question, context)
        words = full_text.split(" ")
        for word in words:
            yield word + " "
            await asyncio.sleep(0.01)


class MockLLMService(BaseLLMService):
    """Mock LLM Service for local testing and CI when API keys are not present."""

    async def generate(self, question: str, context: str) -> str:
        if not context:
            return f"[Offline Mock] I do not have enough context to answer: '{question}'."
        return f"[Offline Mock] Based on the context provided, here is the answer to '{question}'. Context preview: {context[:120]}..."

    async def generate_stream(
        self, question: str, context: str
    ) -> AsyncGenerator[str, None]:
        text = await self.generate(question, context)
        words = text.split(" ")
        for word in words:
            yield word + " "
            await asyncio.sleep(0.02)


def get_llm_service() -> BaseLLMService:
    """Factory dependency returning the configured LLM service."""
    if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip():
        logger.info("Using GeminiLLMService with configured API key.")
        return GeminiLLMService(
            api_key=settings.GEMINI_API_KEY,
            model_name=settings.GEMINI_CHAT_MODEL,
        )
    logger.warning("No GEMINI_API_KEY provided. Defaulting to MockLLMService for offline testing.")
    return MockLLMService()
