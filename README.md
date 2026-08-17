# Smart Chef Agent

Smart Chef is an AI-powered recipe recommendation system built with LangGraph and Groq. It helps users turn leftover or available ingredients into practical, personalized meal suggestions while respecting allergies, dietary preferences, nutrition goals, and ingredient freshness.

The agent can search public recipe data, filter unsafe options, rank the best matches, and then ask the user to choose a recipe before returning full instructions and ingredients.

---

## Overview

Smart Chef is designed for a simple but useful workflow:

1. The user describes what ingredients they have.
2. The agent searches for matching recipes.
3. It filters bad matches based on allergies and dietary restrictions.
4. It optionally considers nutrition goals such as weight loss or weight gain.
5. It prioritizes ingredients that are about to spoil.
6. It ranks the best recommendations and presents them to the user.
7. The user selects a recipe, and the system returns full recipe details.

This project combines:

- LangGraph for the orchestration graph and stateful workflow
- Groq for model inference
- Spoonacular for recipe search and structured recipe details
- Tavily for web-based fallback recipe discovery
- FastAPI for the HTTP API layer
- PostgreSQL support for production persistence via LangGraph checkpointers and stores

---

## Key Features

### Recipe discovery
- Searches for recipes based on ingredients the user currently has.
- Uses Spoonacular as the primary recipe source.
- Falls back to web search through Tavily when Spoonacular results are limited or missing.

### Safety filtering
- Removes recipes that conflict with user allergies.
- Filters out recipes that violate dietary restrictions such as vegan or vegetarian.
- Uses ingredient-based matching rather than relying only on exact recipe names.

### Nutrition-aware ranking
- If the user specifies a goal such as weight loss or weight gain, recipe calorie information is pulled and used to rank results.
- Recipes without calorie data are handled gracefully rather than guessed.

### Freshness prioritization
- Recipes that use perishable ingredients are prioritized to help reduce waste and encourage use of soon-to-expire items.

### Memory and personalization
- Stores durable user profile data such as allergies, dietary restrictions, and goals.
- Reuses preference data across conversation turns without forcing the user to re-enter it every time.

### Interactive recipe selection
- The graph pauses and asks the user to choose a recipe when multiple options are valid.
- Handles both numeric choices and free-text recipe names.

### Streaming and API support
- Provides both standard and streaming endpoints for chat interactions.
- Supports a browser-based client UI for quick testing.

---

## Architecture

The application is built as a LangGraph state machine with a deterministic filter chain and LLM-powered synthesis.

### Core workflow

The graph in `src/graph.py` orchestrates the following phases:

1. `load_profile` loads user profile data from store.
2. `agent` decides whether to search for recipes or skip to profile-only responses.
3. `tools` executes recipe search APIs.
4. `parse_results` converts tool outputs into structured candidate recipes.
5. `allergy_check` removes unsafe recipes.
6. `dietary_check` removes non-compliant recipes.
7. `nutrition_dispatch` and `lookup_nutrition` calculate calorie data when relevant.
8. `freshness_check` prioritizes perishable ingredients.
9. `rank_recipes` sorts and narrows the best matches.
10. `check_results` decides whether to continue searching or synthesize a response.
11. `synthesize` creates a natural-language recommendation summary.
12. `select_recipe` pauses for user confirmation.
13. `get_recipe_details` returns the final full recipe instructions.
14. `extract_profile` persists the updated profile to storage.

---

## Project Structure

```text
smart-chef-agent/
├── api.py                     # FastAPI HTTP API
├── main.py                   # CLI streaming demo
├── langgraph.json            # LangGraph app configuration
├── pyproject.toml            # Python package metadata and dependencies
├── requirements.txt          # Minimal dependency list
├── README.md                 # Project documentation
├── client/
│   └── index.html            # Browser-based frontend
├── src/
│   ├── __init__.py
│   ├── config.py             # Environment configuration
│   ├── filters.py            # Safety checks, ranking, synthesis, nutrition flow
│   ├── graph.py              # LangGraph definition and compilation
│   ├── memory.py             # User profile loading and extraction
│   ├── nodes.py              # Agent node and tool routing
│   ├── recipe_selection.py   # Recipe selection interrupt and full recipe details
│   ├── state.py              # Agent state definition
│   ├── tests/
│   │   └── eval_cases.py     # Evaluation scenarios
│   └── tools/
│       └── recipe_search.py # Spoonacular + Tavily integration
└── smart_chef_agent.egg-info/
```

---

## Technology Stack

- Python 3.12+
- LangGraph
- LangChain
- Groq LLMs
- FastAPI + Uvicorn
- Pydantic Settings
- Spoonacular API
- Tavily API
- PostgreSQL support for production deployment

---

## Environment Setup

Create a `.env` file in the project root with the following values:

```env
GROQ_API_KEY=your_groq_api_key
SPOONACULAR_API_KEY=your_spoonacular_api_key
TAVILY_API_KEY=your_tavily_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
POSTGRES_URI=postgresql://user:password@host:5432/dbname
DEPLOYMENT_MODE=dev
```

### Notes
- `GROQ_API_KEY` is required for LLM calls.
- `SPOONACULAR_API_KEY` is required for recipe search and recipe details.
- `TAVILY_API_KEY` is used for the fallback web search when Spoonacular results are insufficient.
- `LANGSMITH_API_KEY` is used for LangSmith tracing/debugging support.
- `POSTGRES_URI` is only necessary in production mode when using the Postgres-backed checkpointer and store.
- `DEPLOYMENT_MODE` can be set to `dev`, `studio`, or `production`.

---

## Installation

### With pip

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or on Windows:
# .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### With uv (optional)

```bash
uv sync
```

---

## Running the Project

### 1. Run the CLI demo

```bash
python main.py
```

This starts the interactive streaming workflow and lets you test the graph directly from the terminal.

### 2. Run the FastAPI backend

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The server exposes a chat API for frontend or service integration.

### 3. Open the frontend

Open the browser client in `client/index.html` or serve it with a static server if preferred.

### 4. Run with LangGraph Studio / deployment config

```bash
langgraph dev
```

This uses the configuration in `langgraph.json` and the compiled graph named `smart_chef`.

---

## API Endpoints

### POST /chat

Submits a single user message and returns the result of the agent run.

Request body:

```json
{
  "thread_id": "user-thread-1",
  "user_id": "user-123",
  "message": "I have chicken, rice, peanuts, fish and eggs",
  "is_first_turn": true,
  "allergies": ["fish"],
  "dietary_restriction": null,
  "goal": "weight_loss",
  "perishable_ingredients": []
}
```

Response:

```json
{
  "status": "done",
  "final_recommendation": "Here are the best matching recipes...",
  "recipe_details": "Full recipe instructions..."
}
```

If the graph is waiting for the user to pick a recipe, it returns:

```json
{
  "status": "waiting_for_input",
  "prompt": "Which one would you like the full recipe for?"
}
```

### POST /chat/stream

Streams the agent output token-by-token while the graph runs. It also emits an `interrupt` event when the system pauses for recipe selection.

### POST /resume/stream

Used after an interrupt to continue the workflow after the user chooses a recipe.

Request body:

```json
{
  "thread_id": "user-thread-1",
  "user_id": "user-123",
  "reply": "1"
}
```

---

## Example Usage

### Example request to the API

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "demo-thread",
    "user_id": "demo-user",
    "message": "I have chicken, rice, spinach, and eggs",
    "is_first_turn": true,
    "allergies": [],
    "dietary_restriction": "vegetarian",
    "goal": "weight_loss",
    "perishable_ingredients": ["spinach"]
  }'
```

### Example CLI interaction

```text
I have chicken, rice, peanuts, fish and eggs
```

The agent may respond with:

- recipe candidate suggestions
- filtered recommendations that avoid fish
- a prompt asking the user to choose a recipe
- final recipe instructions after they confirm a selection

---

## Behavior and Logic Details

### Filtering pipeline
The app intentionally evaluates recipes in stages so that recommendations are both safe and realistic:

- Search results are parsed into structured recipe objects.
- Allergy matching removes unsafe ingredients.
- Dietary restriction checks exclude incompatible meals.
- Nutrition goals are evaluated only when relevant.
- Perishable ingredients are prioritized to reduce waste.
- Recipes are ranked by ingredient match score and then surfaced to the user.

### Why the system is conservative
The app is designed to avoid guessing. If a recipe is missing data or the user answer is ambiguous, it asks for clarification instead of over-assuming.

Examples include:
- not inventing instructions when Spoonacular does not provide them
- re-prompting when a recipe number or name is not recognized
- avoiding unsafe recipe suggestions when allergy data is present

---

## Deployment Notes

### Local development
- Use `DEPLOYMENT_MODE=dev`.
- The default graph uses `MemorySaver` and `InMemoryStore` so it can run locally without a database.

### Production deployment
When `DEPLOYMENT_MODE=production` is set, the app initializes:

- `ConnectionPool` for PostgreSQL
- `PostgresSaver` for persistent graph checkpoints
- `PostgresStore` for durable cross-session data

This allows the agent to maintain conversation state and user profile information beyond a single in-memory process.

### LangGraph deployment
The project is configured through `langgraph.json` so it can be deployed as a LangGraph app.

---

## Troubleshooting

### Missing environment variables
If the app fails on startup, check that all required environment variables are defined in `.env`.

### Recipe search returns nothing
- Try more generic ingredient names.
- Check whether the request contains ingredients that are not recognized by Spoonacular.
- Use the web-search fallback path when a local dish or traditional recipe is requested.

### API seems to stop before completion
This can happen when the graph triggers an `interrupt` for recipe selection. In that case, the client should call the resume endpoint after receiving the prompt.

---

## License

This project is intended for internal or personal use unless otherwise specified by the repository owner.

---

## Summary

Smart Chef is a practical, multi-step AI recipe assistant that combines structured recipe retrieval, constrained filtering, and user personalization. It is useful for both direct recipe recommendations and interactive cooking assistant workflows, and it is designed to be safe, transparent, and grounded in real data rather than hallucinated cooking advice.
