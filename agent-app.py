# Run with poetry run python -m uvicorn agent-app:app

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any
from dotenv import load_dotenv

from mlflow.types.responses import ResponsesAgentRequest
from mlflow.types.responses_helpers import Message

load_dotenv()
from agent import AGENT

app: FastAPI = FastAPI()


class AgentCallRequest(BaseModel):
    input_string: str


@app.post("/agent-call")
async def agent_call(request: AgentCallRequest) -> dict[str, Any]:
    """
    Receives a string as input and returns a response.
    """
    # Placeholder logic, replace with actual agent call
    return AGENT.predict(
        request=ResponsesAgentRequest(
            input=[Message(role="user", content=request.input_string)]
        )
    ).model_dump_compat()
