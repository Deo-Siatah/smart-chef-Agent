import json
import requests
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send


from src.state import AgentState
from src.config import settings
from langchain_groq import ChatGroq

# Plain (non-tool-bound) LLM, used only for the final synthesis step.
synth_llm = ChatGroq(model="openai/gpt-oss-120b", api_key=settings.GROQ_API_KEY)

DIETARY_BLOCKLISTS = {
    "vegan": {"chicken", "beef", "pork", "fish", "egg", "cheese", "milk", "butter", "honey"},
    "vegetarian": {"chicken", "beef", "pork", "fish"},
}
MAX_RETRIES = 3

ALLERGY_SYNONYMS = {
    "fish": ["fish", "salmon", "tuna", "sardine", "sardines", "mackerel",
             "tilapia", "cod", "anchovy", "anchovies", "omena"],
    "peanut": ["peanut", "peanuts", "groundnut", "groundnuts"],
    "shellfish": ["shrimp", "prawn", "prawns", "crab", "lobster", "shellfish"],
    "dairy": ["milk", "cheese", "butter", "cream", "yogurt", "dairy"],
    "egg": ["egg", "eggs"],
    "tree nut": ["almond", "cashew", "walnut", "pecan", "hazelnut", "tree nut", "nuts"],
}

def _recipe_ingredient_set(recipe: dict) -> set[str]:
    """Helper: lowercase set of all ingredients (used + missing) mentioned in a recipe."""
    used = recipe.get("used_ingredients", [])
    missing = recipe.get("missing_ingredients", [])
    return {i.lower() for i in used + missing}

def _expand_terms(raw_terms: set[str]) -> set[str]:
    """
    Expand user-provided allergy/restriction terms using the synonym
    map, so an allergy of "fish" also catches "sardines", "tilapia",
    etc. — not just recipes that literally say the word "fish".
    """
    expanded = set(raw_terms)
    for term in raw_terms:
        for group_key, synonyms in ALLERGY_SYNONYMS.items():
            if term == group_key or term in synonyms:
                expanded.update(synonyms)
    return expanded
#Stage 0: parse the search tool results out of message history into
# state["candidate_recipes"] as structured data, so the filter stages
# below can operate on real fields instead of re-parsing messages.

def parse_results_node(state: AgentState) -> dict:
    candidates = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            try:
                parsed = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                continue
            for r in parsed:
                candidates.append(
                    {
                        "id": r.get("id"),
                        "name": r.get("name", "Unknown recipe"),
                        "used_ingredients": r.get("used_ingredients", []),
                        "missing_ingredients": r.get("missing_ingredients", []),
                        "source": r.get("source", ""),
                        "snippet": r.get("snippet", ""),
                    }
                )
    return {"candidate_recipes": candidates}

def _contains_match(ingredient_set: set[str], terms: set[str]) -> bool:
    """
    Returns True if any term (e.g. an allergy or blocklisted ingredient)
    appears as a substring of any ingredient name, in either direction.

    Why substring, not exact match: Spoonacular ingredient names are
    often compound — "peanut sauce", "peanut butter", "smoked peanuts" —
    so checking term == ingredient misses almost every real case. We
    check both directions (term in ingredient, and ingredient in term)
    to also catch cases like a user allergy of "tree nuts" matching an
    ingredient literally called "nuts".

    This is intentionally simple substring matching, not stemming or
    fuzzy matching — good enough for common cases, but won't catch
    everything (e.g. "groundnut" won't match "peanut" since they're
    different words for the same thing). Flagging that limitation
    rather than pretending this is bulletproof.
    """
    for term in terms:
        for ingredient in ingredient_set:
            if term in ingredient or ingredient in term:
                return True
    return False


# Stage 1: allergy safety — hard blocker, removes unsafe recipes entirely.

def allergy_check_node(state: AgentState) -> dict:
    allergies = _expand_terms({a.lower() for a in state.get("allergies", [])})
    safe = [
        r for r in state["candidate_recipes"]
        if not _contains_match(_recipe_ingredient_set(r), allergies)
    ]
    return {"candidate_recipes": safe}



def dietary_check_node(state: AgentState) -> dict:
    restriction = state.get("dietary_restriction")
    blocklist = _expand_terms(DIETARY_BLOCKLISTS.get(restriction, set())) if restriction else set()
    compliant = [
        r for r in state["candidate_recipes"]
        if not _contains_match(_recipe_ingredient_set(r), blocklist)
    ]
    return {"candidate_recipes": compliant}


# Stage 3: nutrition — only relevant when the user has a weight goal.
def _get_calories(recipe_id: int) -> int | None:
    """
    Real Spoonacular nutrition lookup. Returns calories as an int,
    or None if the recipe has no id (e.g. came from Tavily web search)
    or the API call fails — callers must handle None gracefully rather
    than assuming every recipe has a calorie value.
    """
    try:
        resp = requests.get(
            f"https://api.spoonacular.com/recipes/{recipe_id}/nutritionWidget.json",
            params={"apiKey": settings.SPOONACULAR_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        # "calories" comes back as a string like "450" from this endpoint
        return int(float(resp.json()["calories"]))
    except (requests.RequestException, KeyError, ValueError):
        return None

from langgraph.types import Send

def nutrition_dispatch_node(state: AgentState):
    """
    Dispatcher: fans out one Send() per recipe that has a Spoonacular id,
    so each recipe's nutrition lookup runs as its own parallel node
    execution instead of one sequential loop. Recipes with no id are
    skipped here (nothing to look up) and simply keep est_calories=None
    when merged back later.
    """
    goal = state.get("goal")
    if goal not in ("weight_loss", "weight_gain"):
        # No goal set — nutrition lookup isn't relevant, skip fan-out
        # entirely and go straight to merge (which becomes a no-op).
        return "merge_nutrition"

    recipes_with_id = [r for r in state["candidate_recipes"] if r.get("id")]
    if not recipes_with_id:
        return "merge_nutrition"

    return [Send("lookup_nutrition", {"recipe": r}) for r in recipes_with_id]


def lookup_nutrition_node(payload: dict) -> dict:
    """
    Worker: runs once per recipe, in parallel with other instances of
    itself. Only sees the ONE recipe it was dispatched with — this is
    the nature of Send() fan-out, each branch is isolated and has no
    visibility into the others.
    """
    recipe = payload["recipe"]
    calories = _get_calories(recipe["id"])
    return {"nutrition_results": [{"id": recipe["id"], "calories": calories}]}


def merge_nutrition_node(state: AgentState) -> dict:
    """
    Join point: runs once, after all parallel nutrition_results have
    been collected (via the operator.add reducer). Matches results
    back onto candidate_recipes by id, then applies the same
    weight-loss/weight-gain sort as before.
    """
    results_by_id = {r["id"]: r["calories"] for r in state.get("nutrition_results", [])}
    recipes = state["candidate_recipes"]

    for r in recipes:
        r["est_calories"] = results_by_id.get(r.get("id"))

    goal = state.get("goal")
    if goal in ("weight_loss", "weight_gain"):
        with_data = [r for r in recipes if r["est_calories"] is not None]
        without_data = [r for r in recipes if r["est_calories"] is None]
        with_data.sort(key=lambda r: r["est_calories"], reverse=(goal == "weight_gain"))
        recipes = with_data + without_data

    # Clear nutrition_results so it doesn't accumulate stale data
    # across retry loops within the same run.
    return {"candidate_recipes": recipes, "nutrition_results": []}

 #Stage 4: freshness — prioritize recipes using perishable ingredients
# first, so leftovers that are about to spoil get used up

def freshness_check_node(state: AgentState) -> dict:
    perishables = {p.lower() for p in state.get("perishable_ingredients", [])}
    recipes = state["candidate_recipes"]

    def uses_perishable(r):
        return bool(_recipe_ingredient_set(r) & perishables)

    # Stable sort: perishable-using recipes bubble to the front,
    # relative order otherwise preserved.
    recipes = sorted(recipes, key=lambda r: not uses_perishable(r))
    return {"candidate_recipes": recipes}


def _match_score(recipe: dict) -> float:
    """
    Fraction of a recipe's ingredients that the user already has on hand.
    1.0 = uses only ingredients they have. Lower = needs more shopping.
    Recipes with no ingredient data at all (e.g. some Tavily results)
    get a neutral 0.5 rather than 0 or 1, so they don't unfairly
    dominate or get buried purely for lacking structured data.
    """
    used = len(recipe.get("used_ingredients", []))
    missing = len(recipe.get("missing_ingredients", []))
    total = used + missing
    if total == 0:
        return 0.5
    return used / total


# How close two scores need to be to count as "basically tied" and
# both get shown to the user, rather than silently picking one.
SCORE_TIE_MARGIN = 0.15
MAX_RECOMMENDATIONS = 3


def rank_recipes_node(state: AgentState) -> dict:
    """
    Scores and sorts candidates by how well they match ingredients the
    user already has, then selects the top 1–3 to present — showing
    more than one only when their scores are close enough to be a
    genuine toss-up, not just padding the list to hit a count.
    """
    recipes = state["candidate_recipes"]
    if not recipes:
        return {}

    for r in recipes:
        r["match_score"] = round(_match_score(r), 2)

    recipes.sort(key=lambda r: r["match_score"], reverse=True)

    top_score = recipes[0]["match_score"]
    selected = [recipes[0]]
    for r in recipes[1:MAX_RECOMMENDATIONS]:
        if top_score - r["match_score"] <= SCORE_TIE_MARGIN:
            selected.append(r)
        else:
            break  # scores are sorted, so once one falls outside the margin, rest will too

    return {"candidate_recipes": selected}

# ---------------------------------------------------------------------
# Final stage: synthesize a natural-language recommendation from
# whatever survived the filter chain.
# ---------------------------------------------------------------------

def synthesize_node(state: AgentState, config: RunnableConfig) -> dict:
    recipes = state["candidate_recipes"]
    if not recipes:
        text = (
            "I couldn't find a recipe that fits all your constraints "
            "(allergies, dietary restriction, or ingredients on hand). "
            "Try loosening one of them or adding more ingredients."
        )
        return {"final_recommendation": text}

    context = json.dumps(recipes, indent=2)

    # Explicitly state what constraints were actually applied, so the
    # model reports on real filtering decisions instead of guessing
    # from what's absent in the recipe list.
    constraints_summary = (
        f"Allergies excluded: {state.get('allergies') or 'none specified'}\n"
        f"Dietary restriction applied: {state.get('dietary_restriction') or 'none specified'}\n"
        f"Goal: {state.get('goal') or 'none specified'}\n"
        f"Perishables prioritized: {state.get('perishable_ingredients') or 'none specified'}"
    )

    prompt = [
        SystemMessage(content=(
            "You are Smart Chef. You will be given the user's actual "
            "constraints (already applied as filters) and 1 to 3 candidate "
            "recipes that already comply with those filters. Present ALL "
            "given recipes as options — do not invent new recipes, do not "
            "suggest substitutions not in the data.\n\n"
            "When explaining why a recipe fits, reference the ACTUAL "
            "constraints given below — never say 'you didn't mention any "
            "allergies' unless the allergies list is genuinely empty. If "
            "an allergy or restriction is listed, say the recipe was "
            "checked against it and avoids it.\n\n"
            "For EACH recipe, state:\n"
            "1. Why it fits their actual allergy/dietary constraints.\n"
            "2. Calories if est_calories is present and how it relates to "
            "their goal; if missing, say plainly it's not available — "
            "never guess a number.\n"
            "3. Which ingredients they already have vs. which are missing."
        )),
        HumanMessage(content=(
            f"User's actual constraints:\n{constraints_summary}\n\n"
            f"Candidate recipes (already filtered to comply):\n{context}\n\n"
            f"User asked: {state['user_input']}"
        )),
    ]
    response = synth_llm.invoke(prompt, config=config)
    return {"final_recommendation": response.content}

# Routing logic — ordered stage list + a helper that finds the next
# APPLICABLE stage, skipping any whose state field is empty/None.
# Adding a new stage later (e.g. repeat-avoidance) means adding one
# tuple here — no other routing code changes.
STAGE_ORDER = [
    ("allergies","allergy_check"),
    ("dietary_restrictions","dietary_check"),
    ("goal","nutrition_check"),
    ("perishable_ingredients","freshness_check"),
]

def _next_applicable(state: AgentState, remaining: list[tuple[str, str]]) -> str:
    for field, node_name in remaining:
        if state.get(field):
            return node_name
        return "synthesize"  # no more applicable stages, go to final synthesis

def route_after_parse(state: AgentState) -> str:
    return _next_applicable(state, STAGE_ORDER)

def route_after_allergy(state: AgentState) -> str:
    return _next_applicable(state, STAGE_ORDER[1:])

def route_after_dietary(state: AgentState) -> str:
    return _next_applicable(state, STAGE_ORDER[2:])

def route_after_nutrition(state: AgentState) -> str:
    return _next_applicable(state, STAGE_ORDER[3:])

def check_results_node(state: AgentState) -> dict:
    """
    Gate node: if no candidates survived the filter chain, either hand
    control back to the agent to try again (up to MAX_RETRIES), or —
    once the cap is hit — force through to synthesize with an apology
    instead of looping indefinitely.
    """
    if state["candidate_recipes"]:
        return {}

    retry_count = state.get("retry_count", 0)

    if retry_count >= MAX_RETRIES:
        # Cap hit — don't increment further, don't loop back to agent.
        # synthesize_node already handles an empty candidate_recipes
        # list gracefully with its own apology message.
        return {}

    reasons = []
    if state.get("allergies"):
        reasons.append(f"allergies: {', '.join(state['allergies'])}")
    if state.get("dietary_restriction"):
        reasons.append(f"dietary restriction: {state['dietary_restriction']}")

    note = (
        "All candidate recipes were filtered out"
        + (f" due to {', and '.join(reasons)}." if reasons else ".")
        + f" (Attempt {retry_count + 1} of {MAX_RETRIES}.) Decide what to do "
          "next: try search_recipes_web with a different or more specific "
          "query, or ask the user a clarifying question if you're out of "
          "good options. Do not just repeat the same search."
    )
    return {
        "messages": [HumanMessage(content=note)],
        "retry_count": retry_count + 1,
    }


def route_after_check(state: AgentState) -> str:
    if state["candidate_recipes"]:
        return "synthesize"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "synthesize"  # cap hit — force through, don't loop back
    return "agent"