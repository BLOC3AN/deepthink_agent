from llms.gemini_llm import LLMGemini
from prompts.build_context import BuildContext
from pydantic import BaseModel, Field
from typing import List, Dict
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

            # Use LLM to analyze user input and create execution plan
            logger.info("🧠 Analyzing user input with LLM for intent detection...")

            # Prepare agents information for LLM
            agents_info = []
            for agent in available_agents:
                agents_info.append({
                    "id": agent.get("id"),
                    "name": agent.get("name"),
                    "type": agent.get("type"),
                    "description": agent.get("description"),
                    "capabilities": agent.get("capabilities", []),
                    "specialization": agent.get("parameters", {}).get("specialization", "")
                })

            # Generate planning using LLM with structured output
            try:
                llm_response = self.model_with_structure.invoke(
                    self.prompt.format_messages(
                        input=user_input,
                        context=str(context or {}),
                        agents=str(agents_info)
                    )
                )

                if llm_response is None:
                    raise Exception("LLM returned None response")

                # Convert to dict
                response = llm_response.model_dump()
                logger.info(f"🎯 LLM detected intents: {response.get('detected_intents', [])}")

            except Exception as llm_error:
                logger.warning(f"LLM structured output failed: {llm_error}, using fallback logic")

                # Fallback: Create plan based on available agents
                response = {
                    "detected_intents": self._detect_intents_fallback(user_input),
                    "execution_strategy": "parallel",
                    "tasks": [],
                    "estimated_time": 75
                }

            # Ensure we have tasks - create them based on available agents if LLM didn't provide
            if not response.get("tasks"):
                logger.info("🔧 Creating tasks based on available agents...")
                response["tasks"] = self._create_tasks_from_agents(available_agents, user_input)

            # Ensure execution strategy is parallel
            response["execution_strategy"] = "parallel"

            logger.info(f"✅ Created {len(response.get('tasks', []))} tasks with phase-based execution")

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

    def _detect_intents_fallback(self, user_input: str) -> List[str]:
        """Fallback intent detection using keyword matching"""
        intents = []
        user_lower = user_input.lower()

        # Intent detection based on keywords
        if any(word in user_lower for word in ["analyze", "analysis", "trend", "data", "performance", "metrics"]):
            intents.append("data_analysis")

        if any(word in user_lower for word in ["summarize", "summary", "brief", "overview", "key points"]):
            intents.append("document_summarization")

        if any(word in user_lower for word in ["validate", "verify", "check", "confirm", "accuracy"]):
            intents.append("data_validation")

        if any(word in user_lower for word in ["market", "business", "industry", "competitive"]):
            intents.append("market_analysis")

        if any(word in user_lower for word in ["technical", "technology", "system", "software"]):
            intents.append("technical_analysis")

        # Default intent if none detected
        if not intents:
            intents.append("general_analysis")

        return intents

    def _create_tasks_from_agents(self, available_agents: List[Dict], user_input: str) -> List[Dict]:
        """Create tasks based on available agents"""
        tasks = []

        # Group agents by type
        analyst_agents = [agent for agent in available_agents if agent.get("type") == "analyst"]
        validation_agents = [agent for agent in available_agents if agent.get("type") == "validation"]
        summary_agents = [agent for agent in available_agents if agent.get("type") == "summary"]

        # Phase 1: Worker agents (parallel)
        if analyst_agents:
            tasks.append({
                "task_id": "worker-analyst",
                "agent_id": analyst_agents[0]["id"],
                "agent_type": "analyst",
                "agent_name": analyst_agents[0]["name"],
                "input_data": user_input,
                "priority": "high",
                "phase": "worker"
            })

        # Phase 2: Validation agent (sequential after workers)
        if validation_agents:
            tasks.append({
                "task_id": "validation",
                "agent_id": validation_agents[0]["id"],
                "agent_type": "validation",
                "agent_name": validation_agents[0]["name"],
                "input_data": user_input,
                "priority": "medium",
                "phase": "validation"
            })

        # Phase 3: Summary agent (sequential after validation)
        if summary_agents:
            tasks.append({
                "task_id": "summary",
                "agent_id": summary_agents[0]["id"],
                "agent_type": "summary",
                "agent_name": summary_agents[0]["name"],
                "input_data": user_input,
                "priority": "low",
                "phase": "summary"
            })

        return tasks
