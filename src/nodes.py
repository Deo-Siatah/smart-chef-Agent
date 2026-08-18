from src.config import settings
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage
from groq import APIError

from src.state import AgentState
from src.tools.recipe_search import search_recipes, search_recipes_web


tools = [search_recipes, search_recipes_web]

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=settings.GROQ_API_KEY, temperature=0)
llm_with_tools = llm.bind_tools(tools)

# System prompt: tells the agent its job and when to use the tool.
SYSTEM_PROMPT = (
    "You are Smart Chef, an assistant that recommends recipes based on "
    "leftover ingredients. When the user tells you what ingredients they "
    "have, use search_recipes (Spoonacular) first. If it returns few or "
    "no relevant results — especially for African or other traditional "
    "cuisine requests — use search_recipes_web instead. Once you have "
    "results, summarize the best recommendation for the user in plain, "
    "friendly language."
)



from langchain_core.runnables import RunnableConfig

def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    messages = state["messages"]
    if not messages or messages[0].type != "system":
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    for attempt in range(2):
        try:
            # config forwarded — this is what lets stream_mode="messages"
            # actually see tokens from this call, instead of only the
            # final AIMessage once invoke() returns.
            response = llm_with_tools.invoke(messages, config=config)
            return {"messages": [response]}
        except APIError:
            if attempt == 0:
                continue
            raise

# src/nodes.py

def route_after_agent(state: AgentState) -> str:
    last_message = state["messages"][-1]

    # Check both the object attribute AND the raw additional_kwargs form —
    # depending on how state was (de)serialized crossing the Studio/API
    # process boundary, only one of these may reliably be populated.
    has_tool_calls = bool(getattr(last_message, "tool_calls", None)) or bool(
        getattr(last_message, "additional_kwargs", {}).get("tool_calls")
    )

    if has_tool_calls:
        return "tools"

    turn_start = state.get("turn_start_index", 0)
    turn_messages = state["messages"][turn_start:]
    searched_this_turn = any(isinstance(m, ToolMessage) for m in turn_messages)
    return "parse_results" if searched_this_turn else "skip_to_profile"

tools_node =ToolNode(tools)