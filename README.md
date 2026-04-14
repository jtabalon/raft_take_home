# Order Parsing Agent

This project implements a Python-based LangGraph agent that:
- accepts a natural-language query
- fetches raw order text from the provided customer API
- parses records with a regex-first strategy
- falls back to OpenRouter `openai/gpt-oss-120b:exacto` for ambiguous records
- validates every parsed record with Pydantic
- applies filtering in Python and returns deterministic JSON

## Architecture
```mermaid
flowchart TD
    user["User query"] --> cli["main.py CLI"]
    cli --> agent["LangGraph OrderAgent"]

    subgraph agent_graph["Agent graph"]
        parse_request["Parse request into OrderQuerySpec"]
        fetch_orders["Fetch raw orders"]
        api_client["CustomerAPIClient"]
        customer_api["Customer orders API"]
        chunk_orders["Chunk orders"]
        regex_parse["Regex parse each record"]
        llm_repair["LLM fallback for ambiguous records"]
        validate["Pydantic validation"]
        filter["Python filtering"]
        format["Format clean JSON response"]

        parse_request --> fetch_orders --> api_client --> customer_api
        customer_api --> chunk_orders --> regex_parse
        regex_parse --> validate
        regex_parse --> llm_repair --> validate
        validate --> filter --> format
    end

    agent --> parse_request
    format --> response["Clean JSON response"]
```

## Setup
1. Install dependencies:
   ```bash
   python3 -m pip install -r requirements.txt
   ```
2. Start the dummy API:
   ```bash
   python3 dummy_customer_api.py
   ```
3. Configure environment:
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your API settings.
   ```

The agent can still run without `OPENROUTER_API_KEY`, but it will use deterministic parsing only and skip LLM fallback.

## Run
```bash
python3 main.py --query "Show me all orders where the buyer was located in Ohio and total value was over 500"
```

## Regression demo
For a small traditional ML baseline, add `--predict-total-for-items` to train an
`sklearn.linear_model.LinearRegression` model on the parsed orders from the
current API response. This is a demonstration feature for the coding challenge,
not a production forecast.

```bash
python3 main.py --query "Show all orders" --predict-total-for-items 2
```

## Run UI
Start the dummy API:
```bash
python3 dummy_customer_api.py
```

In another terminal, start the local UI:
```bash
python3 main.py --ui
```

Open:
```text
http://127.0.0.1:8000
```

The UI also includes an optional "Predict total" field. Enter an item count to
include the same sklearn regression demo in the rendered results and JSON.

## Test
```bash
python3 -m pytest
```

## CI
This repository includes a lightweight GitHub Actions workflow that runs on
pushes and pull requests. The workflow installs dependencies and runs the test
suite with `pytest`, covering the parser, agent behavior, CLI, UI routes, and
regression demo without adding unnecessary deployment infrastructure for this
take-home project.
