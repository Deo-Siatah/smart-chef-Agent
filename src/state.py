from typing import Annotated, TypedDict,Optional
from langgraph.graph.message import add_messages
import operator

class AgentState(TypedDict):
    """
    Represents the state of an agent in the smart chef system.

    """
    user_input: str
    messages: Annotated[list, add_messages]
    ingredients: list[str]
    candidate_recipes: list[dict]
    final_recommendation: str
    allergies: list[str]
    dietary_restrictions: Optional[str]
    goal: Optional[str]
    perishable_ingredients: list[str]
    retry_count: int
    selected_recipe: Optional[dict]   # set once the user picks one, post-interrupt
    recipe_details: str
    turn_start_index: int # index of the first message in the current turn
    nutrition_results: Annotated[list[dict], operator.add]