from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from prompt_optimizer.runner import evaluate


class RunPromptInput(BaseModel):
    """Input schema for RunPrompt."""

    prompt: str = Field(..., description="The prompt to run.")


class RunPrompt(BaseTool):
    name: str = "Run Prompt Tool"
    description: str = "Run a prompt and get the response."
    args_schema: Type[BaseModel] = RunPromptInput

    def _run(self, prompt: str) -> str:
        score, failure_reasons = evaluate(prompt)

        if not failure_reasons:
            return f"Score: {score}"

        failure_text = "\n".join(f"- {reason['reason']}" for reason in failure_reasons)

        return f"""Score: {score}

Failure Reasons:
{failure_text}"""
