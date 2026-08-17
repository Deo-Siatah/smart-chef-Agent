from typing import Optional
from pydantic import BaseModel, Field
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from src.state import AgentState
from src.config import settings

class UserProfile(BaseModel):
    """
    Durable, cross-session facts about a user. This is intentionally a
    SMALL, fixed schema — Trustcall-style extraction works by patching
    specific fields against a known structure, not free-form summarization.
    """
    name: Optional[str] = None
    allergies: list[str] = Field(default_factory=list)
    dietary_restriction: Optional[str] = None
    goal: Optional[str] = None
    disliked_ingredients: list[str] = Field(default_factory=list)
    cuisine_preference: Optional[str] = None


# Structured-output LLM: forces the response to conform to UserProfile's
# schema rather than free text we'd have to parse ourselves.
extraction_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=settings.GROQ_API_KEY,
    temperature=0,
).with_structured_output(UserProfile)

# src/memory.py — updated load_profile_node


def load_profile_node(state: AgentState, config: RunnableConfig, *, store: BaseStore) -> dict:
    updates = {}
    existing_messages = state.get("messages", [])

    if state.get("user_input"):
        updates["messages"] = [HumanMessage(content=state["user_input"])]
        updates["turn_start_index"] = len(existing_messages)
    else:
        updates["turn_start_index"] = max(0, len(existing_messages) - 1)

    user_id = config["configurable"].get("user_id") or "anonymous"
    namespace = (user_id, "profile")
    existing = store.get(namespace, "profile")
    if not existing:
        return updates

    profile = existing.value
    if not state.get("allergies") and profile.get("allergies"):
        updates["allergies"] = profile["allergies"]
    if not state.get("dietary_restriction") and profile.get("dietary_restriction"):
        updates["dietary_restriction"] = profile["dietary_restriction"]
    if not state.get("goal") and profile.get("goal"):
        updates["goal"] = profile["goal"]

    return updates


def extract_profile_node(state: AgentState, config: RunnableConfig, *, store: BaseStore) -> dict:
    user_id = config["configurable"].get("user_id") or "anonymous"
    namespace = (user_id, "profile")
    existing = store.get(namespace, "profile")
    current_profile = existing.value if existing else UserProfile().model_dump()

    turn_start = state.get("turn_start_index", 0)
    turn_messages = state["messages"][turn_start:]
    convo_text = "\n".join(
        f"{m.type}: {m.content}" for m in turn_messages if hasattr(m, "content") and m.content
    )

    explicit_fields = {
        k: v for k, v in {
            "allergies": state.get("allergies"),
            "dietary_restriction": state.get("dietary_restriction"),
            "goal": state.get("goal"),
        }.items() if v
    }

    prompt = [
        SystemMessage(content=(
            "You maintain a user's durable profile for a recipe app. "
            "Given their CURRENT profile, this turn's conversation, and "
            "any explicitly provided fields, return an UPDATED profile. "
            "Only change a field if something clearly indicates a new "
            "value for it — otherwise keep the current value exactly."
        )),
        HumanMessage(content=(
            f"Current profile: {current_profile}\n\n"
            f"This turn's conversation:\n{convo_text}\n\n"
            f"Explicitly provided fields this turn: {explicit_fields or 'none'}"
        )),
    ]

    try:
        updated_profile = extraction_llm.invoke(prompt)
        store.put(namespace, "profile", updated_profile.model_dump())
    except Exception as e:
        # Don't let a flaky extraction call take down the whole turn —
        # the user still gets their answer, profile just doesn't update
        # this time. Print so it's visible in the terminal/Studio logs.
        print(f"[extract_profile_node] extraction failed, profile unchanged: {e}")

    return {}

def skip_to_profile_node(state: AgentState) -> dict:
    """
    Runs when the agent answered directly without searching for recipes
    (e.g. a greeting, a profile update, small talk) — bypasses the
    entire recipe pipeline and uses the agent's own reply as the
    final response.
    """
    last_message = state["messages"][-1]
    return {"final_recommendation": last_message.content}