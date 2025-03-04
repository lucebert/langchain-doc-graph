"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from shared.configuration import BaseConfiguration


@dataclass(kw_only=True)
class RagConfiguration(BaseConfiguration):
    """The configuration for the agent."""

    # models
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o-mini",
        metadata={
            "description": "The language model used for processing and refining queries. Should be in the form: provider/model-name."
        },
    )
