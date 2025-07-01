from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pydantic import BaseModel, Field
from prompt_optimizer.tools.run_prompt import RunPrompt
from prompt_optimizer.tracing import tracer_instance


class EvaluationResult(BaseModel):
    score: float = Field(description="Numerical score of the prompt quality (0-1)")
    failure_reasons: str = Field(description="Summary of failure reasons", default="")


@CrewBase
class PromptEvaluator:
    """PromptEvaluator crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def evaluator(self) -> Agent:
        return Agent(
            config=self.agents_config["evaluator"],
            verbose=True,  # type: ignore[index]
            tools=[RunPrompt()],
        )

    @task
    def evaluate_task(self) -> Task:
        return Task(
            config=self.tasks_config["evaluate_task"],
            output_pydantic=EvaluationResult,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the PromptEvaluator crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
