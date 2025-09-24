from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
import uuid
import asyncio
from core.agent_communicator import AgentCommunicator, TaskRequest, TaskResponse
from database.mongodb_client import MongoDBClient
from utils.logger import Logger

logger = Logger(__name__)


class ExecutionPlan(BaseModel):
    session_id: str
    detected_intents: List[str]
    execution_strategy: str
    tasks: List[Dict]
    estimated_time: int


class AggregatedResult(BaseModel):
    session_id: str
    overall_status: str
    execution_summary: str
    task_results: List[Dict]
    final_output: str
    total_execution_time: float
    timestamp: datetime


class TaskManager:
    """Manages the complete workflow from planning to result aggregation"""
    
    def __init__(self):
        self.communicator = AgentCommunicator()
        self.mongodb = MongoDBClient()
        logger.info("Task Manager initialized")
    
    async def execute_plan(self, execution_plan: ExecutionPlan) -> AggregatedResult:
        """Execute the complete plan and return aggregated results"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Executing plan for session {execution_plan.session_id}")
            
            # Connect to MongoDB
            await self.mongodb.connect()
            
            # Convert plan tasks to task requests
            task_requests = []
            for task in execution_plan.tasks:
                task_request = TaskRequest(
                    task_id=task.get("task_id", str(uuid.uuid4())),
                    agent_type=task.get("agent_type", "summary"),
                    agent_name=task.get("agent_name", "Unknown Agent"),
                    input_data=task.get("input_data", ""),
                    task_context={
                        "session_id": execution_plan.session_id,
                        "intents": execution_plan.detected_intents,
                        "strategy": execution_plan.execution_strategy
                    },
                    priority=task.get("priority", "medium"),
                    timeout=300,
                    phase=task.get("phase", "worker")
                )
                task_requests.append(task_request)
                
                # Create task record in MongoDB
                task_data = {
                    "task_id": task_request.task_id,
                    "session_id": execution_plan.session_id,
                    "agent_type": task_request.agent_type,
                    "status": "pending",
                    "input_data": {
                        "content": task_request.input_data,
                        "context": task_request.task_context,
                        "parameters": {"priority": task_request.priority}
                    },
                    "created_at": datetime.now()
                }
                await self.mongodb.create_task(task_data)
            
            # Group tasks by phase
            worker_tasks = [task for task in task_requests if task.phase == "worker"]
            validation_tasks = [task for task in task_requests if task.phase == "validation"]
            summary_tasks = [task for task in task_requests if task.phase == "summary"]

            all_responses = []

            # Phase 1: Execute worker tasks in parallel (analyst, rag, sql)
            if worker_tasks:
                logger.info(f"✅ Phase 1: Executing {len(worker_tasks)} worker tasks in parallel")
                worker_responses = await self.communicator.execute_tasks_parallel(worker_tasks)
                all_responses.extend(worker_responses)
            else:
                worker_responses = []

            # Phase 2: Execute validation tasks (sequential, with worker results)
            if validation_tasks:
                logger.info(f"✅ Phase 2: Executing {len(validation_tasks)} validation tasks")
                for val_task in validation_tasks:
                    # Prepare validation input with worker results
                    val_task.task_context["worker_results"] = [
                        {"agent_type": r.agent_type, "result": r.result_data}
                        for r in worker_responses
                    ]
                    val_response = await self.communicator.execute_task(val_task)
                    all_responses.append(val_response)

            # Phase 3: Execute summary tasks (sequential, with validation results)
            if summary_tasks:
                logger.info(f"✅ Phase 3: Executing {len(summary_tasks)} summary tasks")
                validation_responses = [r for r in all_responses if r.agent_type == "validation"]
                for sum_task in summary_tasks:
                    # Prepare summary input with all previous results
                    sum_task.task_context["worker_results"] = [
                        {"agent_type": r.agent_type, "result": r.result_data}
                        for r in worker_responses
                    ]
                    sum_task.task_context["validation_results"] = [
                        {"agent_type": r.agent_type, "result": r.result_data}
                        for r in validation_responses
                    ]
                    sum_response = await self.communicator.execute_task(sum_task)
                    all_responses.append(sum_response)

            # All task responses
            task_responses = all_responses
            
            # Update task statuses in MongoDB
            for response in task_responses:
                await self.mongodb.update_task_status(
                    response.task_id,
                    response.status,
                    {
                        "result": response.result_data,
                        "execution_time": response.execution_time,
                        "error_message": response.error_message
                    }
                )
            
            # Aggregate results
            aggregated_result = await self._aggregate_results(
                execution_plan.session_id,
                task_responses,
                start_time
            )
            
            logger.info(f"Plan execution completed for session {execution_plan.session_id}")
            return aggregated_result
            
        except Exception as e:
            logger.error(f"Error executing plan: {e}")
            total_time = (datetime.now() - start_time).total_seconds()
            
            return AggregatedResult(
                session_id=execution_plan.session_id,
                overall_status="failed",
                execution_summary=f"Plan execution failed: {str(e)}",
                task_results=[],
                final_output=f"Execution failed due to error: {str(e)}",
                total_execution_time=total_time,
                timestamp=datetime.now()
            )
        finally:
            await self.mongodb.disconnect()
    
    async def _aggregate_results(self, session_id: str, task_responses: List[TaskResponse], start_time: datetime) -> AggregatedResult:
        """Aggregate task results into final output"""
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Categorize results
        completed_tasks = [r for r in task_responses if r.status == "completed"]
        failed_tasks = [r for r in task_responses if r.status == "failed"]
        timeout_tasks = [r for r in task_responses if r.status == "timeout"]
        
        # Determine overall status
        if len(completed_tasks) == len(task_responses):
            overall_status = "completed"
        elif len(completed_tasks) > 0:
            overall_status = "partial_success"
        else:
            overall_status = "failed"
        
        # Create execution summary
        execution_summary = f"Executed {len(task_responses)} tasks: {len(completed_tasks)} completed, {len(failed_tasks)} failed, {len(timeout_tasks)} timed out"
        
        # Generate final output from phase results
        final_output_parts = []

        # Summary results (final phase output)
        summary_results = [r for r in completed_tasks if r.agent_type == "summary"]
        if summary_results:
            for result in summary_results:
                if "summary" in result.result_data:
                    final_output_parts.append(result.result_data["summary"])

        # If no summary, use validation results
        if not final_output_parts:
            validation_results = [r for r in completed_tasks if r.agent_type == "validation"]
            if validation_results:
                for result in validation_results:
                    if "validation_summary" in result.result_data:
                        final_output_parts.append(result.result_data["validation_summary"])

        # If no validation, use analysis results
        if not final_output_parts:
            analysis_results = [r for r in completed_tasks if r.agent_type == "analyst"]
            if analysis_results:
                for result in analysis_results:
                    if "executive_summary" in result.result_data:
                        final_output_parts.append(result.result_data["executive_summary"])

        final_output = "\n".join(final_output_parts) if final_output_parts else "No results generated"

        # Add error information if any
        if failed_tasks or timeout_tasks:
            final_output += "\n\n## Issues Encountered\n"
            for task in failed_tasks + timeout_tasks:
                final_output += f"- {task.agent_type} agent: {task.error_message}\n"
        
        # Prepare task results for response
        task_results = []
        for response in task_responses:
            task_results.append({
                "task_id": response.task_id,
                "agent_type": response.agent_type,
                "agent_name": response.agent_name,
                "status": response.status,
                "execution_time": response.execution_time,
                "result_summary": self._summarize_result(response.result_data),
                "error_message": response.error_message
            })
        
        return AggregatedResult(
            session_id=session_id,
            overall_status=overall_status,
            execution_summary=execution_summary,
            task_results=task_results,
            final_output=final_output,
            total_execution_time=total_time,
            timestamp=datetime.now()
        )
    
    def _summarize_result(self, result_data: Dict) -> str:
        """Create a brief summary of result data"""
        if not result_data:
            return "No result data"
        
        if "error" in result_data:
            return f"Error: {result_data['error']}"
        
        # Summary agent result
        if "summary" in result_data:
            return f"Summary: {result_data['summary'][:100]}..."
        
        # Analyst agent result
        if "executive_summary" in result_data:
            return f"Analysis: {result_data['executive_summary'][:100]}..."
        
        # Validation agent result
        if "overall_status" in result_data:
            return f"Validation: {result_data['overall_status']}"
        
        return "Result generated successfully"


