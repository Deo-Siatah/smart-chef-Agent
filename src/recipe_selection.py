# src/recipe_selection.py
import re
from langgraph.types import interrupt
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from src.state import AgentState
from src.tools.recipe_search import get_recipe_information
from src.filters import synth_llm  # reuse the same plain LLM used for synthesis


def _match_selection(user_text: str, recipes: list[dict]) -> dict | None:
    """
    Match free text or a numbered choice against the recipe list.
    Tries, in order: a leading digit ("1", "2. Kale Rolls"), then a
    substring match against recipe names (case-insensitive). Returns
    None if nothing matches confidently, so the caller can re-ask
    rather than guess.
    """
    text = user_text.strip().lower()

    # Numbered choice: "1", "2.", "option 3", etc.
    digit_match = re.search(r"\d+", text)
    if digit_match:
        idx = int(digit_match.group()) - 1
        if 0 <= idx < len(recipes):
            return recipes[idx]

    # Free text: substring match against recipe name
    for r in recipes:
        if r["name"].lower() in text or text in r["name"].lower():
            return r

    return None


def select_recipe_node(state: AgentState) -> dict:
    """
    Pauses the graph and waits for the user to pick one of the
    presented recipes (by number or by name). Re-prompts on an
    unrecognized answer rather than guessing, since picking the
    wrong recipe here means the wrong nutrition/allergy info gets
    presented as if it were verified.
    """
    recipes = state["candidate_recipes"]

    if len(recipes) == 1:
        # Only one option — still confirm rather than silently proceeding,
        # since the user may want to back out entirely.
        prompt = f"I'd recommend {recipes[0]['name']}. Want the full recipe? (yes/no)"
    else:
        options = "\n".join(f"{i+1}. {r['name']}" for i, r in enumerate(recipes))
        prompt = f"Which one would you like the full recipe for?\n{options}"

    user_reply = interrupt(prompt)

    if len(recipes) == 1:
        if str(user_reply).strip().lower() not in ("yes", "y"):
            return {"selected_recipe": None}
        return {"selected_recipe": recipes[0]}

    matched = _match_selection(str(user_reply), recipes)
    if matched is None:
        # No confident match — interrupt again with the same prompt.
        # (LangGraph will re-run this node from the top on resume.)
        retry_reply = interrupt(f"Sorry, I didn't catch that. {prompt}")
        matched = _match_selection(str(retry_reply), recipes)

    return {"selected_recipe": matched}


def get_recipe_details_node(state: AgentState, config:RunnableConfig) -> dict:
    """
    Produces the full step-by-step recipe: real ingredient quantities
    and instructions when Spoonacular has them, explicit 'not available'
    when it doesn't — never fabricated specifics.
    """
    recipe = state.get("selected_recipe")
    if not recipe:
        return {"recipe_details": "No recipe selected — let me know if you'd like to pick one after all."}

    full_info = get_recipe_information(recipe["id"]) if recipe.get("id") else None

    if full_info:
        data_block = (
            f"Servings: {full_info['servings']}\n"
            f"Ready in: {full_info['ready_in_minutes']} minutes\n"
            f"Ingredients (with quantities): {full_info['ingredients']}\n"
            f"Instructions (raw): {full_info['instructions']}"
        )
        instruction_note = "Use the real data below — do not invent quantities or steps."
    else:
        data_block = (
            f"No structured recipe data available for '{recipe['name']}'. "
            f"Source link: {recipe.get('source', 'none')}\n"
            f"Snippet: {recipe.get('snippet', 'none')}"
        )
        instruction_note = (
            "No verified quantities or steps are available. Say this "
            "plainly and point the user to the source link rather than "
            "inventing a step-by-step recipe."
        )

    goal = state.get("goal")
    calorie_note = (
        f"Estimated calories: {recipe.get('est_calories')}"
        if recipe.get("est_calories") is not None
        else "Calorie data not available — say so plainly, do not guess."
    )

    prompt = [
        SystemMessage(content=(
            "You are Smart Chef. Present the full recipe the user selected. "
            f"{instruction_note} If the user has a weight goal ({goal or 'none'}), "
            "reference the calorie note explicitly. Format as: prep time/servings "
            "(if known), ingredient list with quantities, numbered steps."
        )),
        HumanMessage(content=f"{data_block}\n\n{calorie_note}"),
    ]
    response = synth_llm.invoke(prompt, config=config)  # forward config so stream_mode works
    return {"recipe_details": response.content}