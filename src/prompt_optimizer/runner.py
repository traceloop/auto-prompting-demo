from typing import List, Dict, Optional
from prompt_optimizer.rag import query_rag
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
from rich import print as rprint
from rich.console import Console

client = OpenAI()
console = Console()

MAX_EVALUATION_EXAMPLES: Optional[int] = 2

evaluation_items = [
    {
        "question": "Can I self-host Traceloop?",
        "required_facts": [
            "Self-hosting is possible",
            "Docker deployment option",
            "Local installation requirements",
        ],
    },
    {
        "question": "How do I start using Traceloop?",
        "required_facts": [
            "SDK installation process",
            "Initial configuration steps",
            "Basic usage example",
        ],
    },
    {
        "question": "What programming languages does Traceloop support?",
        "required_facts": [
            "Python support",
            "JavaScript/TypeScript support",
            "Other language through Hub",
        ],
    },
    {
        "question": "What kind of metrics can I track with Traceloop?",
        "required_facts": [
            "LLM metrics",
            "Vector DB metrics",
            "Custom metrics support",
        ],
    },
    {
        "question": "Is Traceloop suitable for microservices architecture?",
        "required_facts": [
            "Microservices architecture compatibility",
            "Distributed tracing support",
            "Scalability features",
        ],
    },
    {
        "question": "How does Traceloop handle security and data privacy?",
        "required_facts": [
            "Data encryption",
            "Privacy controls",
            "Compliance standards",
        ],
    },
    {
        "question": "What are the system requirements for running Traceloop?",
        "required_facts": [
            "Hardware requirements",
            "Software dependencies",
            "Network requirements",
        ],
    },
    {
        "question": "Can I integrate Traceloop with my existing monitoring stack?",
        "required_facts": [
            "Datadog integration",
            "Grafana integration",
            "Other monitoring tools compatibility",
        ],
    },
    {
        "question": "What kind of support and documentation is available for Traceloop?",
        "required_facts": [
            "Documentation availability",
            "Support channels",
            "Community resources",
        ],
    },
    {
        "question": "What is OpenLLMetry and how does it work?",
        "required_facts": [
            "OpenLLMetry definition",
            "Core functionality",
            "Key features",
        ],
    },
    {
        "question": "How do I initialize OpenLLMetry in my Python application?",
        "required_facts": [
            "SDK installation",
            "Initialization code",
            "Basic configuration",
        ],
    },
    {
        "question": "What frameworks does OpenLLMetry support?",
        "required_facts": [
            "Framework compatibility list",
            "Integration methods",
            "Version requirements",
        ],
    },
    {
        "question": "How do I use OpenLLMetry with LangChain?",
        "required_facts": [
            "LangChain integration steps",
            "Required components",
            "Usage examples",
        ],
    },
    {
        "question": "What observability platforms can I export OpenLLMetry traces to?",
        "required_facts": [
            "Jaeger support",
            "Zipkin support",
            "Other platform compatibility",
        ],
    },
    {
        "question": "How do I track user feedback with OpenLLMetry?",
        "required_facts": [
            "Feedback collection methods",
            "Integration points",
            "Data storage",
        ],
    },
    {
        "question": "How do I manually report LLM and Vector DB calls in OpenLLMetry?",
        "required_facts": [
            "Manual reporting API",
            "Vector DB tracking",
            "LLM call tracking",
        ],
    },
    {
        "question": "What are the privacy considerations when using OpenLLMetry?",
        "required_facts": [
            "Data handling policies",
            "User data protection",
            "Compliance measures",
        ],
    },
    {
        "question": "How do I handle workflow annotations in OpenLLMetry?",
        "required_facts": [
            "Annotation syntax",
            "Workflow definition",
            "Integration points",
        ],
    },
    {
        "question": "Can I use OpenLLMetry without the SDK?",
        "required_facts": ["SDK alternatives", "Manual implementation", "Limitations"],
    },
    {
        "question": "How do I trace user IDs with traces in OpenLLMetry?",
        "required_facts": [
            "User ID tracking methods",
            "Trace correlation",
            "Privacy considerations",
        ],
    },
    {
        "question": "What are the supported integrations for OpenLLMetry?",
        "required_facts": [
            "Supported platforms",
            "Integration methods",
            "Version compatibility",
        ],
    },
    {
        "question": "How do I handle versioning in OpenLLMetry?",
        "required_facts": [
            "Version management",
            "Upgrade process",
            "Backward compatibility",
        ],
    },
    {
        "question": "What are the best practices for using OpenLLMetry with threads in Python?",
        "required_facts": [
            "Thread safety features",
            "Best practices",
            "Common pitfalls",
        ],
    },
]


class FactEvaluation(BaseModel):
    fact: str
    passed: bool
    reason: str


class ResponseEvaluation(BaseModel):
    fact_evaluations: List[FactEvaluation]


def evaluate_single_fact(question: str, response: str, fact: str) -> FactEvaluation:
    prompt = f"""You are an evaluator checking if a specific fact is present in an answer.
    
Question: {question}

Fact to check: {fact}

Answer to evaluate:
{response}

Determine if this specific fact is present in the answer. Consider both explicit mentions and implicit coverage.
Provide a clear reason for your decision.
"""

    result = client.responses.parse(
        model="gpt-4o",
        input=[
            {
                "role": "system",
                "content": "Evaluate if the specific fact is present in the answer.",
            },
            {"role": "user", "content": prompt},
        ],
        text_format=FactEvaluation,
    )

    return result.output_parsed


def evaluate_single_response(
    question: str, response: str, required_facts: List[str]
) -> ResponseEvaluation:
    fact_evaluations = []
    passed_count = 0

    for fact in required_facts:
        evaluation = evaluate_single_fact(question, response, fact)
        fact_evaluations.append(evaluation)
        if evaluation.passed:
            passed_count += 1

    return ResponseEvaluation(fact_evaluations=fact_evaluations)


def process_and_evaluate_single_question(
    prompt_template: str, item: Dict
) -> Dict[str, any]:
    question = item["question"]
    response = query_rag(prompt_template, question)

    evaluation = evaluate_single_response(question, response, item["required_facts"])
    passed_count = sum(1 for eval in evaluation.fact_evaluations if eval.passed)

    return {
        "question": question,
        "response": response,
        "score": passed_count / len(item["required_facts"]),
        "fact_evaluations": [eval.model_dump() for eval in evaluation.fact_evaluations],
    }


def evaluate(prompt_template: str):
    rprint(
        f"[bold blue]🔍[/bold blue] [bold green]Evaluating prompt:[/bold green] [yellow]{prompt_template}[/yellow]"
    )

    total_passed = 0
    total_facts = 0
    evaluated_responses = []
    failure_reasons = []

    items_to_evaluate = (
        evaluation_items[:MAX_EVALUATION_EXAMPLES]
        if MAX_EVALUATION_EXAMPLES
        else evaluation_items
    )

    with tqdm(
        total=len(items_to_evaluate),
        desc="Progress",
        position=0,
        colour="green",
        bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt}",
    ) as pbar_questions:
        with tqdm(
            total=0,
            desc="Current Score",
            position=1,
            bar_format="{desc}: {n_fmt}/{total_fmt} {postfix}",
        ) as pbar_facts:
            with tqdm(
                total=0,
                desc="Current Question:",
                position=2,
                bar_format="{desc}",
                postfix="",
            ) as pbar_question:
                for item in items_to_evaluate:
                    result = process_and_evaluate_single_question(prompt_template, item)
                    evaluated_responses.append(result)

                    passed_count = sum(
                        1 for eval in result["fact_evaluations"] if eval["passed"]
                    )
                    total_passed += passed_count
                    total_facts += len(item["required_facts"])

                    for eval in result["fact_evaluations"]:
                        if not eval["passed"]:
                            failure_reasons.append(
                                {
                                    "question": item["question"],
                                    "fact": eval["fact"],
                                    "reason": eval["reason"],
                                }
                            )

                    pbar_questions.update(1)
                    pbar_facts.total = total_facts
                    pbar_facts.n = total_passed
                    pbar_facts.set_postfix_str(f"({total_passed/total_facts:.2f})")
                    pbar_question.set_description_str(
                        f"Current Question: {item['question']}"
                    )
                    pbar_facts.refresh()
                    pbar_question.refresh()

    total_score = sum(result["score"] for result in evaluated_responses) / len(
        evaluated_responses
    )

    return total_score, failure_reasons


def run():
    prompt_template = """Answer the following question based on the provided context:
Context:
{context}

Question:
{question}"""
    total_score, failure_reasons = evaluate(prompt_template)
    print(f"\nOverall Score: {total_score:.2f}")

    if failure_reasons:
        print("\nAnalyzing failure patterns...")
        failure_summary = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "Analyze the following failure reasons and provide a concise summary of the main patterns and issues.",
                },
                {
                    "role": "user",
                    "content": f"Here are the failure reasons:\n{str(failure_reasons)}",
                },
            ],
        )
        print("\nFailure Analysis:")
        print(failure_summary.choices[0].message.content)
