from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
from typing import List


@CrewBase
class PromptOptimizer:
    """PromptOptimizer crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            verbose=True,  # type: ignore[index]
        )

    @agent
    def prompt_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["prompt_engineer"],  # type: ignore[index]
            verbose=True,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],  # type: ignore[index]
        )

    @task
    def improve_prompt_task(self) -> Task:
        return Task(
            config=self.tasks_config["improve_prompt_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the PromptOptimizer crew"""

        prompt_optimization_papers = CrewDoclingSource(
            file_paths=[
                "https://arxiv.org/pdf/2401.14423",
            ],
        )

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            knowledge_sources=[prompt_optimization_papers],
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
