"""
tools/llm_factory.py

Central LLM factory – returns the correct LangChain chat model
based on config.provider and the associated API key.

Supported providers:
  - google   → ChatGoogleGenerativeAI  (Gemini)
  - openai   → ChatOpenAI              (GPT)
  - anthropic → ChatAnthropic          (Claude)
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from ..config import config


def get_llm(creative: bool = False) -> BaseChatModel:
    """
    Return a configured chat LLM for the current provider.

    Parameters
    ----------
    creative : bool
        If True, uses the higher temperature (llm_temperature_creative),
        used by the Insight Generator for synthesis.
    """
    temperature = config.llm_temperature_creative if creative else config.llm_temperature
    model = config.llm_model
    provider = config.provider.lower()

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=config.api_key,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=config.api_key,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            anthropic_api_key=config.api_key,
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            "Choose from: 'google', 'openai', 'anthropic'."
        )
