import json
from typing import Any, Callable, Generator
import os
from uuid import uuid4
import logging

import backoff
import mlflow
import openai
from openai.types.responses import FunctionToolParam

from mlflow.entities import SpanType
from mlflow.pyfunc.model import ResponsesAgent
from mlflow.types.responses_helpers import ResponseOutputItemDoneEvent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from openai import OpenAI
from pydantic import BaseModel


OUTPUT_ITEM_DONE = "response.output_item.done"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolInfo(BaseModel):
    """
    Class representing a tool for the agent.
    - "name" (str): The name of the tool.
    - "spec" (dict): JSON description of the tool (matches OpenAI Responses format)
    - "exec_fn" (Callable): Function that implements the tool logic
    """

    name: str
    spec: FunctionToolParam
    exec_fn: Callable


class ToolCallingAgentNoMemory(ResponsesAgent):
    """
    Class representing a tool-calling Agent
    """
    client: OpenAI
    _tools_dict: dict[str, ToolInfo]
    model: str

    def __init__(self, base_url: str, api_key: str, model: str, tools: list[ToolInfo]):
        """Initializes the ToolCallingAgent with tools."""
        logger.info("Initializing ToolCallingAgentNoMemory with model: %s", model)
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self._tools_dict = {tool.name: tool for tool in tools}
        self.model = model

    def get_tool_specs(self) -> list[FunctionToolParam]:
        """Returns tool specifications in the format OpenAI expects."""
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    #@mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Executes the specified tool with the given arguments."""
        return self._tools_dict[tool_name].exec_fn(**args)

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    # @mlflow.trace(span_type=SpanType.LLM)
    def call_llm(self, input_messages) -> ResponsesAgentStreamEvent:
        logger.info("Calling LLM with messages: %s", input_messages)
        response = self.client.responses.create(
            model=self.model,
            input=input_messages,
            tools=self.get_tool_specs(),
        )
        output = response.output[0]
        logger.info("LLM output: %s", output)
        return ResponsesAgentStreamEvent(
            type=OUTPUT_ITEM_DONE,
            item=output.model_dump(exclude_none=True)
        )

    def handle_tool_call(self, tool_call: dict[str, Any]) -> ResponsesAgentStreamEvent:
        """
        Execute tool calls and return a ResponsesAgentStreamEvent w/ tool output
        """
        args = json.loads(tool_call["arguments"])
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))

        tool_call_output = {
            "type": "function_call_output",
            "call_id": tool_call["call_id"],
            "output": result,
        }
        return ResponsesAgentStreamEvent(
            type=OUTPUT_ITEM_DONE,
            item=tool_call_output,
        )

    def call_and_run_tools(
        self,
        input_messages: list[dict[str, str]],
        max_iter: int = 10,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        mlflow.log_text("Starting tool-calling agent loop.", "log.txt")
        mlflow.log_text("Initial input messages: " + str(input_messages), "log.txt")
        for _ in range(max_iter):
            last_msg = input_messages[-1]
            logger.info("Last message type: %s", last_msg.get("type", None))
            if (
                last_msg.get("type", None) == "message"
                and last_msg.get("role", None) == "assistant"
            ):
                return
            if last_msg.get("type", None) == "function_call":
                tool_call_res = self.handle_tool_call(last_msg)
                output = tool_call_res.custom_outputs
                output_item = output.get("item") if output else None
                mlflow.log_text("Tool call output: " + str(output_item), "log.txt")
                if output_item:
                    input_messages.append(output_item)
                yield tool_call_res
            else:
                llm_output = self.call_llm(input_messages=input_messages)
                if llm_output.item:
                    input_messages.append(llm_output.item)
                yield llm_output

        yield ResponsesAgentStreamEvent(
            type=OUTPUT_ITEM_DONE,
            item={
                "id": str(uuid4()),
                "content": [
                    {
                        "type": "output_text",
                        "text": "Max iterations reached. Stopping.",
                    }
                ],
                "role": "assistant",
                "type": "message",
            }
        )

    #@mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            ResponseOutputItemDoneEvent(**event.model_dump_compat()).item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(
            output=outputs, custom_outputs=request.custom_inputs
        )

    #@mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        input_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            i.model_dump() for i in request.input
        ]
        yield from self.call_and_run_tools(input_messages=input_messages)


tools = [
    ToolInfo(
        name="get_time",
        spec={
            "type": "function",
            "name": "get_time",
            "description": "Get current time for the provided time zone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Name of the time zone."},
                },
                "required": ["timezone"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        exec_fn=lambda timezone: __import__("datetime").datetime.now(__import__("pytz").timezone(timezone)).isoformat()
    )
]

SYSTEM_PROMPT = "You are a helpful assistant that can call tools to get information."
mlflow.openai.autolog()
AGENT = ToolCallingAgentNoMemory(
    base_url=os.environ["BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
    model=os.environ["MODEL"],
    tools=tools)
mlflow.models.set_model(AGENT)