from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional
from utils.logger import Logger
from agents.summaryAgent.summary_agent import SummaryAgent
import os 
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = Logger(__name__)

app = FastAPI(title="Agent Hub", version="1.0.0")

class TaskRequest(BaseModel):
    task_id: str
    agent_type: str
    agent_name: str
    input_data: str
    context: str


class TaskResponse(BaseModel):
    task_id: str
    agent_type: str
    agent_name: str
    status: str  # "completed", "failed", "timeout"
    result_data: Dict
    execution_time: float
    error_message: Optional[str] = None


@app.get("/agent_cards")
async def root():
    """Get list of available agent cards"""
    agent_cards = []
    cards_folder = os.path.join(os.path.dirname(__file__), "schema")
    logger.info(cards_folder)
    for filename in os.listdir(cards_folder):
        if filename.endswith(".json"):
            with open(os.path.join(cards_folder, filename), "r") as f:
                agent_cards.append(json.load(f))
    return agent_cards


@app.post("/task/summary", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    """Execute a task on the appropriate agent"""
    import time
    start_time = time.time()

    try:
        logger.info(f"Received task request: {request.task_id}")

        # Run the agent
        input_context = f"maintask: {request.input_data} context: {request.context}"
        agent_result = SummaryAgent().run(input_context)
        execution_time = time.time() - start_time

        # Convert agent result to TaskResponse format
        if "error" in agent_result:
            response = TaskResponse(
                task_id=request.task_id,
                agent_type=request.agent_type,
                agent_name=request.agent_name,
                status="failed",
                result_data=agent_result,
                execution_time=execution_time,
                error_message=agent_result.get("error", "Unknown error")
            )
        else:
            response = TaskResponse(
                task_id=request.task_id,
                agent_type=request.agent_type,
                agent_name=request.agent_name,
                status="completed",
                result_data=agent_result,
                execution_time=execution_time
            )

        logger.info(f"Task completed: {response.status}")
        return response

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error executing task: {e}")

        error_response = TaskResponse(
            task_id=request.task_id,
            agent_type=request.agent_type,
            agent_name=request.agent_name,
            status="failed",
            result_data={"error": str(e)},
            execution_time=execution_time,
            error_message=str(e)
        )
        return error_response
