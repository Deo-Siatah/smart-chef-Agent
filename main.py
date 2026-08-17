from langgraph.types import Command
from langchain_core.messages import HumanMessage
from src.graph import graph

def run_turn_streaming(thread_id: str, user_id: str, user_text: str, is_first_turn: bool = False, **profile_fields):
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    if is_first_turn:
        state_update = {
            "user_input": user_text,
            "ingredients": [],
            "candidate_recipes": [],
            "final_recommendation": "",
            "retry_count": 0,
            "selected_recipe": None,
            "recipe_details": "",
            "nutrition_results": [],
            **profile_fields,
        }
    else:
        state_update = {"user_input": user_text}

    # stream_mode=["updates", "messages"] gives us BOTH kinds at once:
    # "updates" events tell us which node just ran and what it changed;
    # "messages" events give us LLM tokens as they're generated.
    for mode, chunk in graph.stream(state_update, config=config, stream_mode=["updates", "messages"]):
        if mode == "updates":
            # chunk is {node_name: {state_updates}} — use this for
            # coarse progress indicators.
            for node_name in chunk:
                if node_name not in ("__interrupt__",):
                    print(f"[{node_name}] ...")
        elif mode == "messages":
            # chunk is (message_chunk, metadata) — use this for
            # token-by-token output, but only from nodes whose text
            # we actually want to show live (synthesize, get_recipe_details).
            message_chunk, metadata = chunk
            if metadata.get("langgraph_node") in ("synthesize", "get_recipe_details"):
                if message_chunk.content:
                    print(message_chunk.content, end="", flush=True)

    # Handle interrupt: graph.stream() raises/pauses similarly to invoke();
    # simplest correct approach is to fall back to invoke() once interrupted,
    # since resuming a stream mid-interrupt needs the same Command pattern.
    final_state = graph.get_state(config)
    while final_state.next and "select_recipe" in final_state.next:
        prompt_text = final_state.tasks[0].interrupts[0].value
        print(f"\n{prompt_text}")
        reply = input("> ")
        for mode, chunk in graph.stream(Command(resume=reply), config=config, stream_mode=["updates", "messages"]):
            if mode == "messages":
                message_chunk, metadata = chunk
                if metadata.get("langgraph_node") == "get_recipe_details" and message_chunk.content:
                    print(message_chunk.content, end="", flush=True)
        final_state = graph.get_state(config)

    print()  # newline after streamed output


if __name__ == "__main__":
    run_turn_streaming(
        "deo-stream-test", "deo",
        "I have chicken, rice, peanuts,fish and eggs",
        is_first_turn=True,
        allergies=["fish"], dietary_restriction=None,
        goal="weight_loss", perishable_ingredients=[],
    )