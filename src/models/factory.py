from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from src.models.settings import load_agent_model_settings


@lru_cache(maxsize=None)
def get_chat_model(provider: str, model: str, temperature: float) -> BaseChatModel:
    """Create a chat model through LangChain so agents stay provider-agnostic."""
    return init_chat_model(
        model,
        model_provider=provider,
        temperature=temperature,
    )


def get_agent_model(agent_name: str) -> BaseChatModel:
    """Return the chat model configured for a named agent."""
    settings = load_agent_model_settings(agent_name)
    return get_chat_model(settings.provider, settings.model, settings.temperature)
