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
                    timeout=300
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
            
            # Separate worker tasks from aggregation tasks
            worker_tasks = [task for task in task_requests if task.agent_type != "aggregation"]
            aggregation_tasks = [task for task in task_requests if task.agent_type == "aggregation"]

            # Execute worker tasks in parallel
            if worker_tasks:
                logger.info(f"Executing {len(worker_tasks)} worker tasks in parallel")
                worker_responses = await self.communicator.execute_tasks_parallel(worker_tasks)
            else:
                worker_responses = []

            # Execute aggregation tasks after worker tasks complete
            aggregation_responses = []
            if aggregation_tasks:
                logger.info(f"Executing {len(aggregation_tasks)} aggregation tasks")
                for agg_task in aggregation_tasks:
                    # Prepare aggregation input with worker results
                    agg_task.input_data = self._prepare_aggregation_data(worker_responses)
                    agg_response = await self.communicator.execute_task(agg_task)
                    aggregation_responses.append(agg_response)

            # Combine all responses
            task_responses = worker_responses + aggregation_responses
            
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
        
        # Check if we have aggregation results
        aggregation_results = [r for r in completed_tasks if r.agent_type == "aggregation"]

        if aggregation_results:
            # Use aggregation result as primary output
            agg_result = aggregation_results[0].result_data
            if "final_summary" in agg_result:
                final_output = agg_result["final_summary"]

                # Add additional sections from aggregation
                if "key_insights" in agg_result and agg_result["key_insights"]:
                    final_output += "\n\n## Key Insights\n"
                    for insight in agg_result["key_insights"]:
                        final_output += f"- {insight}\n"

                if "recommendations" in agg_result and agg_result["recommendations"]:
                    final_output += "\n## Recommendations\n"
                    for rec in agg_result["recommendations"]:
                        final_output += f"- {rec}\n"

                if "confidence_score" in agg_result:
                    confidence = agg_result["confidence_score"]
                    final_output += f"\n## Confidence Score: {confidence:.1%}\n"
            else:
                final_output = "Aggregation completed but no summary available"
        else:
            # Fallback to individual agent results
            final_output_parts = []

            # Add summary results
            summary_results = [r for r in completed_tasks if r.agent_type == "summary"]
            if summary_results:
                final_output_parts.append("## Summary Results")
                for result in summary_results:
                    if "summary" in result.result_data:
                        final_output_parts.append(f"- {result.result_data['summary']}")

            # Add analysis results
            analysis_results = [r for r in completed_tasks if r.agent_type == "analyst"]
            if analysis_results:
                final_output_parts.append("\n## Analysis Results")
                for result in analysis_results:
                    if "executive_summary" in result.result_data:
                        final_output_parts.append(f"- {result.result_data['executive_summary']}")
                    if "recommendations" in result.result_data:
                        final_output_parts.append("### Recommendations:")
                        for rec in result.result_data["recommendations"]:
                            final_output_parts.append(f"  - {rec}")

            # Add validation results
            validation_results = [r for r in completed_tasks if r.agent_type == "validation"]
            if validation_results:
                final_output_parts.append("\n## Validation Results")
                for result in validation_results:
                    if "overall_status" in result.result_data:
                        final_output_parts.append(f"- Validation Status: {result.result_data['overall_status']}")
                    if "validation_summary" in result.result_data:
                        final_output_parts.append(f"- {result.result_data['validation_summary']}")

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

    def _prepare_aggregation_data(self, worker_responses: List[TaskResponse]) -> str:
        """Prepare aggregation input data from worker task responses"""
        aggregation_data = []

        for response in worker_responses:
            task_data = {
                "agent_type": response.agent_type,
                "agent_name": response.agent_name,
                "status": response.status,
                "execution_time": response.execution_time,
                "result_data": response.result_data,
                "error_message": response.error_message
            }
            aggregation_data.append(task_data)

        return str(aggregation_data)
