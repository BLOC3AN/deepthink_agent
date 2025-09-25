from llms.gemini_llm import LLMGemini
from prompts.build_context import BuildContext
from pydantic import BaseModel, Field
from typing import List, Dict
from utils.logger import Logger
import uuid

logger = Logger(__name__)


class TaskAssignment(BaseModel):
    task_id: str = Field(..., description="Unique task ID")
    agent_name: str = Field(..., description="Selected agent name from agent cards")
    agent_url: str = Field(..., description="Agent URL for communication")
    agent_type: str = Field(..., description="Type of agent: summary, analyst, validation")
    input_data: str = Field(..., description="description of the task")
    context : str = Field(..., description="Context for the task, specific to the task, reason for task creation")


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

    def _get_agent_cards(self) -> List[Dict]:
        import requests
        return requests.get("http://127.0.0.1:8000/agent_cards").json()


    def run(self, user_input: str):
        logger.info("Planning Agent starting analysis")

        try:
            # Use mock agents for testing
            available_agents = self._get_agent_cards()
            logger.info(f"Using {len(available_agents)} mock agents for testing")

            if not available_agents:
                logger.warning("No active agents found")
                return {"error": "No active agents available"}

            # Use LLM to analyze user input and create execution plan
            logger.info("🧠 Analyzing user input with LLM for intent detection...")

            # Prepare agents information for LLM
            agents_info = []
            for agent in available_agents:
                agents_info.append({
                    "name": agent.get("name"),
                    "agent_type": agent.get("type"),
                    "description": agent.get("description"),
                    "capabilities": agent.get("capabilities", []),
                    "url": agent.get("url"),
                    "status": agent.get("status"),
                    "method": agent.get("method"),
                    "arg_schema": agent.get("arg_schema", {})
                })

            # Generate planning using LLM with structured output
            llm_response = self.model_with_structure.invoke(
                self.prompt.format_messages(
                    input=user_input,
                    agents=str(agents_info)
                )
            )

            if llm_response is None:
                raise Exception("LLM returned None response")

            # Convert to dict
            response = llm_response.model_dump()
            logger.info(f"🎯 LLM detected intents: {response.get('detected_intents', [])}")
            logger.info(f"🎯 LLM created {len(response.get('tasks', []))} tasks")

            # Ensure execution strategy is parallel
            response["execution_strategy"] = "parallel"

            logger.info(f"✅ LLM planning completed with {len(response.get('tasks', []))} tasks")

            # Create session record (mock for testing)
            session_id = str(uuid.uuid4())
            response["session_id"] = session_id

            logger.info(f"Planning completed with {len(response.get('tasks', []))} tasks")
            return response

        except Exception as e:
            logger.error(f"Error in planning agent: {e}")
            return {"error": str(e)}
