from llms.gemini_llm import LLMGemini
from prompts.build_context import BuildContext
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from database.mongodb_client import MongoDBClient
from utils.logger import Logger
import uuid
from datetime import datetime

logger = Logger(__name__)


class TaskAssignment(BaseModel):
    task_id: str = Field(..., description="Unique task ID")
    agent_name: str = Field(..., description="Selected agent name from agent cards")
    agent_url: str = Field(..., description="Agent URL for communication")
    agent_type: str = Field(..., description="Type of agent: summary, analyst, validation")
    input_data: str = Field(..., description="Data to be processed by agent")
    priority: str = Field(..., description="Task priority: high, medium, low")


class PlanningOutput(BaseModel):
    detected_intents: List[str] = Field(..., description="List of detected intents from user input")
    execution_strategy: str = Field(..., description="Overall execution strategy")
    tasks: List[TaskAssignment] = Field(..., description="List of task assignments for agents")
    estimated_time: int = Field(..., description="Estimated execution time in seconds")


class PlanningAgent:
    def __init__(self):
        self.llm = LLMGemini()
        self.prompt = BuildContext().context_planning("prompts/tasks/planning.yml")
        self.model_with_structure = self.llm.model.with_structured_output(PlanningOutput)
        self.mongodb = MongoDBClient()

    async def run(self, user_input: str, context: Dict = None):
        logger.info("Planning Agent starting analysis")

        try:
            # Connect to MongoDB and get available agents
            await self.mongodb.connect()
            available_agents = await self.mongodb.get_all_agent_cards()

            if not available_agents:
                logger.warning("No active agents found in database")
                return {"error": "No active agents available"}

            # Planning Agent analyzes and selects agents
            planning_input = {
                "user_input": user_input,
                "context": context or {},
                "available_agents": available_agents
            }

            # LLM analyzes intent and selects appropriate agents
            llm_response = self.model_with_structure.invoke(
                self.prompt.format_messages(
                    input=user_input,
                    context=str(context or {}),
                    agents=str(available_agents)
                )
            )

            if llm_response is None:
                raise Exception("LLM returned None response")

            response = llm_response.model_dump()

            # Always add aggregation task if we have worker tasks
            worker_tasks = [task for task in response.get("tasks", []) if task.get("agent_type") != "aggregation"]
            if worker_tasks:
                # Find aggregation agent
                aggregation_agents = [agent for agent in available_agents if agent.get("type") == "aggregation"]
                if aggregation_agents:
                    agg_agent = aggregation_agents[0]
                    aggregation_task = {
                        "task_id": str(uuid.uuid4()),
                        "agent_id": agg_agent["id"],
                        "agent_type": "aggregation",
                        "agent_name": agg_agent["name"],
                        "input_data": "Worker task results will be provided",
                        "priority": "high"
                    }
                    response["tasks"].append(aggregation_task)
                    logger.info("Added aggregation task to execution plan")

            # Force parallel execution strategy
            response["execution_strategy"] = "parallel"

            # Create session record
            session_data = {
                "session_id": str(uuid.uuid4()),
                "user_input": user_input,
                "context": context or {},
                "detected_intents": response.get("detected_intents", []),
                "execution_plan": {
                    "tasks": [task["task_id"] for task in response.get("tasks", [])],
                    "estimated_time": response.get("estimated_time", 0)
                },
                "status": "planning",
                "created_at": datetime.now()
            }

            session_id = await self.mongodb.create_session(session_data)
            response["session_id"] = session_id

            logger.info(f"Planning completed with {len(response.get('tasks', []))} tasks")
            return response

        except Exception as e:
            logger.error(f"Error in planning agent: {e}")
            return {"error": str(e)}
        finally:
            await self.mongodb.disconnect()

