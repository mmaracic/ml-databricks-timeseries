import json
from typing import Any, Callable, Generator
import os
from uuid import uuid4
import logging
from datetime import datetime

import requests

import backoff
import mlflow
import openai
from openai.types.responses import FunctionToolParam

from mlflow.entities import SpanType
from mlflow.pyfunc.model import ResponsesAgent
from mlflow.types.responses_helpers import (
    ResponseOutputItemDoneEvent,
    FunctionCallOutput,
)
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from openai import OpenAI
from openai.types.responses import ResponseOutputItem
from pydantic import BaseModel


OUTPUT_ITEM_DONE = "response.output_item.done"
FUNCTION_CALL_DONE = "response.function_call_arguments.done"
LOG_FILE = "log.txt"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolInfo(BaseModel):
    """
    Class representing a tool for the agent.
    - "spec" (dict): JSON description of the tool (matches OpenAI Responses format)
    - "exec_fn" (Callable): Function that implements the tool logic
    """

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
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self._tools_dict = {tool.spec["name"]: tool for tool in tools}
        self.model = model

    def get_tool_specs(self) -> list[FunctionToolParam]:
        """Returns tool specifications in the format OpenAI expects."""
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    # @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Executes the specified tool with the given arguments."""
        return self._tools_dict[tool_name].exec_fn(**args)

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    # @mlflow.trace(span_type=SpanType.LLM)
    def call_llm(
        self, input_messages: list[dict[str, str]]
    ) -> list[ResponseOutputItem]:
        logger.info("Calling LLM with messages: %s", input_messages)
        response = self.client.responses.create(
            model=self.model,
            input=",".join(
                [str(msg) for msg in input_messages]
            ),  # Its critical to convert messages to a single string
            tools=self.get_tool_specs(),
        )
        output = response.output
        logger.info("LLM output: %s", output)
        return output

    def handle_tool_call(self, tool_call: dict[str, Any]) -> FunctionCallOutput:
        """
        Execute tool calls and return a ResponsesAgentStreamEvent w/ tool output
        """
        args = json.loads(tool_call["arguments"])
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))

        return FunctionCallOutput(
            call_id=tool_call["call_id"],
            output=result,
            status="completed",
        )  # These params are required to be compatible with OpenAI input type FunctionCallOutput

    def call_and_run_tools(
        self,
        input_messages: list[dict[str, str]],
        max_iter: int = 10,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        mlflow.log_text("Starting tool-calling agent loop.", LOG_FILE)
        mlflow.log_text("Initial input messages: " + str(input_messages), LOG_FILE)

        for _ in range(max_iter):
            last_msg = input_messages[-1]
            last_msg_type = last_msg.get("type", None)
            last_msg_role = last_msg.get("role", None)
            logger.info(
                "Last message type: %s and role: %s", last_msg_type, last_msg_role
            )
            if last_msg_type == "message" and last_msg_role == "assistant":
                logger.info("Last message is a final assistant output: %s", last_msg)
                return
            if last_msg_type == "function_call":
                logger.info("Handling tool call: %s", last_msg)
                tool_call_res = self.handle_tool_call(last_msg)
                logger.info("Tool call output: %s", tool_call_res)
                input_messages.append(tool_call_res.model_dump())
                yield ResponsesAgentStreamEvent(
                    type=FUNCTION_CALL_DONE,
                    item=tool_call_res.model_dump(exclude_none=True),
                )
            else:
                llm_output = self.call_llm(input_messages=input_messages)
                last_msg = llm_output[-1].model_dump(exclude_none=True)
                input_messages.extend([item.model_dump() for item in llm_output])
                yield ResponsesAgentStreamEvent(
                    type=OUTPUT_ITEM_DONE,
                    item=last_msg,
                )

        last_msg = input_messages[-1]
        last_msg_type = last_msg.get("type", None)
        last_msg_role = last_msg.get("role", None)
        if not (last_msg_type == "message" and last_msg_role == "assistant"):
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
                },
            )

    # @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            ResponseOutputItemDoneEvent(**event.model_dump()).item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(
            output=outputs, custom_outputs=request.custom_inputs
        )

    # @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        input_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            i.model_dump() for i in request.input
        ]
        yield from self.call_and_run_tools(input_messages=input_messages)


tools = [
    ToolInfo(
        spec={
            "type": "function",
            "name": "get_date_time",
            "description": "Get current date and time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
        exec_fn=lambda: "Current date and time is: " + datetime.now().isoformat(),
    ),
    ToolInfo(
        spec={
            "type": "function",
            "name": "get_sales_prediction",
            "description": "Get predicted sales for defined start date and end date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "string",
                        "description": "Start date for the sales prediction in the format YYYY-MM-DD.",
                    },
                    "end": {
                        "type": "string",
                        "description": "End date for the sales prediction in the format YYYY-MM-DD.",
                    },
                },
                "required": ["start", "end"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        exec_fn=lambda start, end: requests.post(
            "https://dbc-7d1169bb-4536.cloud.databricks.com/serving-endpoints/arima-0-0/invocations",
            headers={"Content-Type": "application/json"},
            auth=("token", os.environ["DATABRICKS_API_TOKEN"]),
            data={
                "dataframe_split": {"columns": ["start", "end"], "data": [[start, end]]}
            },
            timeout=60,
        ).text,
    ),
]

SYSTEM_PROMPT = "You are a helpful assistant that can call tools to get information."
mlflow.openai.autolog()
AGENT = ToolCallingAgentNoMemory(
    base_url=os.environ["BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
    model=os.environ["MODEL"],
    tools=tools,
)
mlflow.models.set_model(AGENT)
