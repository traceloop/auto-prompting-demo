from typing import Optional

from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel

from prompt_optimizer.evaluate_crew.evaluate_crew import (
    PromptEvaluator,
    EvaluationResult,
)
from prompt_optimizer.optimize_crew.optimize_crew import PromptOptimizer


START_PROMPT = """Answer the following question based on the provided context:
Context:
{context}

Question:
{question}"""


class PromptOptimizationFlowState(BaseModel):
    prompt: str = START_PROMPT
    feedback: Optional[str] = None
    valid: bool = False
    retry_count: int = 0
    score: float = 0.0


class PromptOptimizationFlow(Flow[PromptOptimizationFlowState]):

    @start("retry")
    def evaluate_prompt(self):
        print("Evaluating prompt")
        result: EvaluationResult = (
            PromptEvaluator().crew().kickoff(inputs={"prompt": self.state.prompt})
        ).pydantic
        self.state.score = result.score
        self.state.valid = not result.failure_reasons
        self.state.feedback = result.failure_reasons

        print(f"Evaluation results:")
        print(f"Score: {self.state.score:.2f}")
        if result.failure_reasons:
            print("\nFailure reasons:")
            print(result.failure_reasons)

        self.state.retry_count += 1

        return "optimize"

    @router(evaluate_prompt)
    def optimize_prompt(self):
        if self.state.score > 0.8:
            return "complete"

        if self.state.retry_count > 3:
            return "max_retry_exceeded"

        print("Optimizing prompt")
        result = (
            PromptOptimizer()
            .crew()
            .kickoff(
                inputs={
                    "prompt": self.state.prompt,
                    "feedback": self.state.feedback,
                    "score": self.state.score,
                }
            )
        )

        print("Optimized prompt:", result.raw)
        self.state.prompt = result.raw

        return "retry"

    @listen("complete")
    def save_result(self):
        print("Prompt is valid")
        print(f"Final prompt (Score: {self.state.score:.2f}):")
        print(self.state.prompt)

        with open("optimized_prompt.txt", "w") as file:
            file.write(f"Score: {self.state.score:.2f}\n")
            file.write(f"Prompt:\n{self.state.prompt}")

    @listen("max_retry_exceeded")
    def max_retry_exceeded_exit(self):
        print("Max retry count exceeded")
        print(f"Final prompt (Score: {self.state.score:.2f}):")
        print(self.state.prompt)
        if self.state.feedback:
            print("\nRemaining failure reasons:")
            print(self.state.feedback)


def kickoff():
    prompt_flow = PromptOptimizationFlow()
    prompt_flow.kickoff()


def plot():
    prompt_flow = PromptOptimizationFlow()
    prompt_flow.plot()


if __name__ == "__main__":
    kickoff()
