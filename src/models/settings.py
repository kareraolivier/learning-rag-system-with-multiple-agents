import os
from dataclasses import dataclass

DEFAULT_PROVIDER = os.getenv("DEFAULT_MODEL_PROVIDER", "ollama")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5:3b")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_MODEL_TEMPERATURE", "0"))

AGENT_ENV_PREFIXES = {
    "query_generator": "QUERY_GENERATOR",
    "answer_generator": "ANSWER_GENERATOR",
    "translator": "TRANSLATOR",
}


@dataclass(frozen=True)
class AgentModelSettings:
    provider: str
    model: str
    temperature: float


def load_agent_model_settings(agent_name: str) -> AgentModelSettings:
    """Read provider/model/temperature for one agent from the environment."""
    try:
        prefix = AGENT_ENV_PREFIXES[agent_name]
    except KeyError as exc:
        known = ", ".join(sorted(AGENT_ENV_PREFIXES))
        raise ValueError(f"Unknown agent '{agent_name}'. Expected one of: {known}") from exc

    return AgentModelSettings(
        provider=os.getenv(f"{prefix}_MODEL_PROVIDER", DEFAULT_PROVIDER),
        model=os.getenv(f"{prefix}_MODEL", DEFAULT_MODEL),
        temperature=float(os.getenv(f"{prefix}_MODEL_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
    )
