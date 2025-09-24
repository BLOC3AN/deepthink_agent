from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
import uuid
import asyncio
from utils.logger import Logger

logger = Logger(__name__)


class TaskRequest(BaseModel):
    task_id: str
    agent_type: str
    agent_name: str
    input_data: str
    task_context: Dict = {}
    priority: str = "medium"
    timeout: int = 300


class TaskResponse(BaseModel):
    task_id: str
    agent_type: str
    agent_name: str
    status: str  # "completed", "failed", "timeout"
    result_data: Dict
    execution_time: float
    error_message: Optional[str] = None
    timestamp: datetime


class AgentCommunicator:
    """Handles communication between Planning Agent and worker agents"""
    
    def __init__(self):
        self.active_tasks = {}
        logger.info("Agent Communicator initialized")
    
    async def execute_task(self, task_request: TaskRequest) -> TaskResponse:
        """Execute a single task on the appropriate agent"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Executing task {task_request.task_id} on {task_request.agent_type} agent")
            
            # Import and instantiate the appropriate agent
            agent = await self._get_agent_instance(task_request.agent_type)
            
            if not agent:
                raise Exception(f"Unknown agent type: {task_request.agent_type}")
            
            # Execute the task
            result = await agent.run(
                input_data=task_request.input_data,
                task_context=task_request.task_context
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TaskResponse(
                task_id=task_request.task_id,
                agent_type=task_request.agent_type,
                agent_name=task_request.agent_name,
                status="completed",
                result_data=result,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except asyncio.TimeoutError:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Task {task_request.task_id} timed out after {execution_time}s")
            
            return TaskResponse(
                task_id=task_request.task_id,
                agent_type=task_request.agent_type,
                agent_name=task_request.agent_name,
                status="timeout",
                result_data={},
                execution_time=execution_time,
                error_message=f"Task timed out after {task_request.timeout} seconds",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Task {task_request.task_id} failed: {e}")
            
            return TaskResponse(
                task_id=task_request.task_id,
                agent_type=task_request.agent_type,
                agent_name=task_request.agent_name,
                status="failed",
                result_data={},
                execution_time=execution_time,
                error_message=str(e),
                timestamp=datetime.now()
            )
    
    async def execute_tasks_parallel(self, task_requests: List[TaskRequest]) -> List[TaskResponse]:
        """Execute multiple tasks in parallel"""
        logger.info(f"Executing {len(task_requests)} tasks in parallel")
        
        # Create tasks for parallel execution
        tasks = []
        for task_request in task_requests:
            task = asyncio.create_task(
                asyncio.wait_for(
                    self.execute_task(task_request),
                    timeout=task_request.timeout
                )
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle exception as failed task
                task_request = task_requests[i]
                responses.append(TaskResponse(
                    task_id=task_request.task_id,
                    agent_type=task_request.agent_type,
                    agent_name=task_request.agent_name,
                    status="failed",
                    result_data={},
                    execution_time=0.0,
                    error_message=str(result),
                    timestamp=datetime.now()
                ))
            else:
                responses.append(result)
        
        logger.info(f"Parallel execution completed. {len([r for r in responses if r.status == 'completed'])} succeeded, {len([r for r in responses if r.status != 'completed'])} failed")
        return responses
    
    async def _get_agent_instance(self, agent_type: str):
        """Get agent instance based on type"""
        try:
            if agent_type == "summary":
                from agents.summary_agent import SummaryAgent
                return SummaryAgent()
            elif agent_type == "analyst":
                from agents.analyst_agent import AnalystAgent
                return AnalystAgent()
            elif agent_type == "validation":
                from agents.validation_agent import ValidationAgent
                return ValidationAgent()
            elif agent_type == "aggregation":
                from agents.aggregation_agent import AggregationAgent
                return AggregationAgent()
            else:
                logger.error(f"Unknown agent type: {agent_type}")
                return None
        except ImportError as e:
            logger.error(f"Failed to import agent {agent_type}: {e}")
            return None
