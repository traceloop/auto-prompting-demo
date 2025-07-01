from typing import Optional

from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel

from prompt_optimizer.evaluate_crew.evaluate_crew import (
    PromptEvaluator,
    EvaluationResult,
)
from prompt_optimizer.optimize_crew.optimize_crew import PromptOptimizer
from prompt_optimizer.tracing import tracer_instance


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
        with tracer_instance.start_as_current_span("evaluate_prompt") as span:
            print("Evaluating prompt")
            span.set_attribute("prompt", self.state.prompt)
            span.set_attribute("retry_count", self.state.retry_count)
            
            result: EvaluationResult = (
                PromptEvaluator().crew().kickoff(inputs={"prompt": self.state.prompt})
            ).pydantic
            
            self.state.score = result.score
            self.state.valid = not result.failure_reasons
            self.state.feedback = result.failure_reasons

            span.set_attribute("score", self.state.score)
            span.set_attribute("valid", self.state.valid)
            if result.failure_reasons:
                span.set_attribute("failure_reasons", result.failure_reasons)

            print(f"Evaluation results:")
            print(f"Score: {self.state.score:.2f}")
            if result.failure_reasons:
                print("\nFailure reasons:")
                print(result.failure_reasons)

            self.state.retry_count += 1

            return "optimize"

    @router(evaluate_prompt)
    def optimize_prompt(self):
        with tracer_instance.start_as_current_span("optimize_prompt") as span:
            span.set_attribute("current_score", self.state.score)
            span.set_attribute("retry_count", self.state.retry_count)
            
            if self.state.score > 0.8:
                span.set_attribute("optimization_result", "complete")
                return "complete"

            if self.state.retry_count > 3:
                span.set_attribute("optimization_result", "max_retry_exceeded")
                return "max_retry_exceeded"

            print("Optimizing prompt")
            span.set_attribute("original_prompt", self.state.prompt)
            span.set_attribute("feedback", self.state.feedback or "")
            
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
            span.set_attribute("optimized_prompt", result.raw)
            span.set_attribute("optimization_result", "retry")
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
    with tracer_instance.start_as_current_span("prompt_optimization_flow") as span:
        span.set_attribute("operation", "kickoff")
        prompt_flow = PromptOptimizationFlow()
        prompt_flow.kickoff()


def plot():
    prompt_flow = PromptOptimizationFlow()
    prompt_flow.plot()


if __name__ == "__main__":
    kickoff()
