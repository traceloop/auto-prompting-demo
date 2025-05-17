# Prompt Optimizer Crew

Welcome to the PromptOptimizer Crew project, powered by [crewAI](https://crewai.com). This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

## Installation

Ensure you have Python >=3.10 <3.13 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

```bash
uv install
```

Then, you can run the following commands:

- `uv run load_data` - Prepares and loads data for RAG operations. This must be run first before using other commands
- `uv run rag` - Runs the RAG pipeline once and stops
- `uv run kickoff` - Attempts to optimize the prompt used for RAG by running multiple iterations
