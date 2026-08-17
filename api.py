"""
Thin HTTP wrapper around the compiled graph. Run with:
    DEPLOYMENT_MODE=production uvicorn api:app --reload
"""
import os
os.environ.setdefault("DEPLOYMENT_MODE", "production")  

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from src.graph import graph

app = FastAPI(title="Smart Chef API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class ChatRequest(BaseModel):
    thread_id: str
    user_id: str
    message: str
    is_first_turn: bool = False
    allergies: list[str] | None = None
    dietary_restriction: str | None = None
    goal: str | None = None
    perishable_ingredients: list[str] | None = None

class ResumeRequest(BaseModel):
    thread_id: str
    user_id: str
    reply: str

def _config(req_thread_id: str, req_user_id: str) -> dict:
    return {"configurable": {"thread_id": req_thread_id, "user_id": req_user_id}}

@app.post("/chat")
def chat(req: ChatRequest):
    """
    Single-turn, non-streaming endpoint. If the graph pauses on the
    select_recipe interrupt, this returns the interrupt prompt instead
    of a final answer — the client is expected to call /resume next.
    """
    config = _config(req.thread_id, req.user_id)

    state_update = {"user_input": req.message}
    if req.is_first_turn:
        state_update.update({
            "ingredients": [], "candidate_recipes": [], "final_recommendation": "",
            "retry_count": 0, "selected_recipe": None, "recipe_details": "",
            "nutrition_results": [],
            "allergies": req.allergies or [],
            "dietary_restriction": req.dietary_restriction,
            "goal": req.goal,
            "perishable_ingredients": req.perishable_ingredients or [],
        })
    result = graph.invoke(state_update, config=config)

    if "__interrupt__" in result:
        return {"status": "waiting_for_input", "prompt": result["__interrupt__"][0].value}

    return {
        "status": "done",
        "final_recommendation": result.get("final_recommendation", ""),
        "recipe_details": result.get("recipe_details", ""),
    }

@app.post("/resume/stream")
def resume_stream(req: ResumeRequest):
    config = _config(req.thread_id, req.user_id)

    def event_generator():
        for mode, chunk in graph.stream(Command(resume=req.reply), config=config, stream_mode=["messages"]):
            message_chunk, metadata = chunk
            if metadata.get("langgraph_node") == "get_recipe_details" and message_chunk.content:
                yield f"data: {message_chunk.content}\n\n"
        yield "event: done\ndata: \n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    config = _config(req.thread_id, req.user_id)
    state_update = {"user_input": req.message}
    if req.is_first_turn:
        state_update.update({
            "ingredients": [], "candidate_recipes": [], "final_recommendation": "",
            "retry_count": 0, "selected_recipe": None, "recipe_details": "",
            "nutrition_results": [],
            "allergies": req.allergies or [],
            "dietary_restriction": req.dietary_restriction,
            "goal": req.goal,
            "perishable_ingredients": req.perishable_ingredients or [],
        })

    def event_generator():
        for mode, chunk in graph.stream(state_update, config=config, stream_mode=["messages", "updates"]):
            if mode == "messages":
                message_chunk, metadata = chunk
                if metadata.get("langgraph_node") in ("synthesize", "get_recipe_details"):
                    if message_chunk.content:
                        yield f"data: {message_chunk.content}\n\n"
            elif mode == "updates" and "__interrupt__" in chunk:
                # Distinct event TYPE (not just "data") so the frontend
                # can tell "here's the recipe picker prompt" apart from
                # normal streamed answer text, and switch UI modes.
                prompt_text = chunk["__interrupt__"][0].value
                yield f"event: interrupt\ndata: {prompt_text}\n\n"
                return  # stream ends here — frontend must call /resume next

        yield "event: done\ndata: \n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")