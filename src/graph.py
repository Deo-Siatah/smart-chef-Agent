import os
from psycopg_pool import ConnectionPool
from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from src.memory import load_profile_node, extract_profile_node, skip_to_profile_node
from src.recipe_selection import select_recipe_node, get_recipe_details_node
from src.config import settings

from src.state import AgentState
from src.nodes import agent_node, tools_node, route_after_agent
from src.filters import (
    parse_results_node,
    allergy_check_node,
    dietary_check_node,
    freshness_check_node,
    nutrition_dispatch_node,
    lookup_nutrition_node,
    merge_nutrition_node,
    rank_recipes_node,
    check_results_node,
    synthesize_node,
    route_after_parse,
    route_after_allergy,
    route_after_dietary,
    route_after_nutrition,
    route_after_check,
)

builder = StateGraph(AgentState)

# --- Step 1 nodes ---
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)

# --- Step 2 nodes ---
builder.add_node("parse_results", parse_results_node)
builder.add_node("allergy_check", allergy_check_node)
builder.add_node("dietary_check", dietary_check_node)

# "nutrition_dispatch" is a NO-OP node — it exists only so we have a
# source node to attach the fan-out conditional edge to. The actual
# fan-out logic lives in nutrition_dispatch_node, used below as a
# ROUTING function, not as this node's body.
builder.add_node("nutrition_dispatch", lambda state: {})
builder.add_node("lookup_nutrition", lookup_nutrition_node)
builder.add_node("merge_nutrition", merge_nutrition_node)

builder.add_node("freshness_check", freshness_check_node)
builder.add_node("rank_recipes", rank_recipes_node)
builder.add_node("check_results", check_results_node)
builder.add_node("synthesize", synthesize_node)

builder.add_node("select_recipe", select_recipe_node)
builder.add_node("get_recipe_details", get_recipe_details_node)

builder.add_node("load_profile", load_profile_node)
builder.add_node("extract_profile", extract_profile_node)
builder.add_node("skip_to_profile", skip_to_profile_node)

# --- edges ---
builder.add_edge(START, "load_profile")
builder.add_edge("load_profile", "agent")

builder.add_conditional_edges(
    "agent", route_after_agent,
    {"tools": "tools", "parse_results": "parse_results", "skip_to_profile": "skip_to_profile"},
)
builder.add_edge("tools", "agent")
builder.add_edge("skip_to_profile", "extract_profile")

builder.add_conditional_edges(
    "parse_results", route_after_parse,
    {"allergy_check": "allergy_check", "dietary_check": "dietary_check",
     "nutrition_check": "nutrition_dispatch", "freshness_check": "freshness_check",
     "synthesize": "rank_recipes"},
)
builder.add_conditional_edges(
    "allergy_check", route_after_allergy,
    {"dietary_check": "dietary_check", "nutrition_check": "nutrition_dispatch",
     "freshness_check": "freshness_check", "synthesize": "rank_recipes"},
)
builder.add_conditional_edges(
    "dietary_check", route_after_dietary,
    {"nutrition_check": "nutrition_dispatch", "freshness_check": "freshness_check",
     "synthesize": "rank_recipes"},
)

# Fan-out: nutrition_dispatch_node used HERE as the routing function —
# it inspects state and returns either "merge_nutrition" (skip case)
# or a list of Send(...) objects, one per recipe needing a lookup.
builder.add_conditional_edges(
    "nutrition_dispatch", nutrition_dispatch_node,
    ["lookup_nutrition", "merge_nutrition"],
)
builder.add_edge("lookup_nutrition", "merge_nutrition")

# merge_nutrition takes over nutrition_check's old position in the chain
builder.add_conditional_edges(
    "merge_nutrition", route_after_nutrition,
    {"freshness_check": "freshness_check", "synthesize": "rank_recipes"},
)

builder.add_edge("freshness_check", "rank_recipes")
builder.add_edge("rank_recipes", "check_results")

builder.add_conditional_edges(
    "check_results", route_after_check,
    {"synthesize": "synthesize", "agent": "agent"},
)

builder.add_edge("synthesize", "select_recipe")
builder.add_edge("select_recipe", "get_recipe_details")
builder.add_edge("get_recipe_details", "extract_profile")
builder.add_edge("extract_profile", END)

DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "dev")  # "dev" | "studio" | "production"

if DEPLOYMENT_MODE == "studio" or os.getenv("LANGGRAPH_API_URL") or os.getenv("LANGGRAPH_AUTH_TYPE"):
    graph = builder.compile()

elif DEPLOYMENT_MODE == "production":
    # Use psycopg_pool for Neon Postgres
    db_uri = settings.POSTGRES_URI  # e.g. postgres://user:pass@host/db?sslmode=require
    pool = ConnectionPool(db_uri, min_size=1, max_size=10, timeout=30)

    # Pass the pool into LangGraph’s PostgresSaver and PostgresStore
    checkpointer = PostgresSaver(pool)
    store = PostgresStore(pool)

    # Ensure tables exist
    checkpointer.setup()
    store.setup()

    # Compile graph with pooled connections
    graph = builder.compile(checkpointer=checkpointer, store=store)

else:
    # Local dev fallback
    graph = builder.compile(checkpointer=MemorySaver(), store=InMemoryStore())