# OpenLLMetry Tracing Setup

This application now includes OpenLLMetry tracing to monitor LLM calls and performance in Traceloop.

## Setup

1. **Install dependencies** (if not already installed):
   ```bash
   uv sync
   ```

2. **Set your Traceloop API key** (optional but recommended):
   ```bash
   export TRACELOOP_API_KEY="your_api_key_here"
   ```
   
   You can get your API key from your [Traceloop dashboard](https://app.traceloop.com/).

3. **Run the application**:
   ```bash
   uv run kickoff
   ```

## What Gets Traced

The application automatically traces:

- **Main prompt optimization flow**: Full workflow execution
- **Prompt evaluation**: Each evaluation step with scores and feedback
- **Prompt optimization**: Optimization attempts with before/after prompts
- **Retry logic**: Retry counts and completion reasons

## Trace Attributes

Each trace includes relevant attributes such as:
- Prompt content (original and optimized)
- Evaluation scores
- Retry counts
- Failure reasons
- Optimization results

## Viewing Traces

1. Go to your [Traceloop dashboard](https://app.traceloop.com/)
2. Navigate to the "Traces" section
3. View detailed traces of your prompt optimization runs

## Local Development

If you don't set the `TRACELOOP_API_KEY` environment variable, the application will still run and trace locally, but traces won't be sent to Traceloop. This allows for development without requiring API keys.

## Environment Variables

- `TRACELOOP_API_KEY`: Your Traceloop API key (optional for local development)