from llms.gemini_llm import LLMGemini
from prompts.build_context import BuildContext
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from utils.logger import Logger

logger = Logger(__name__)


class AggregationAgentOutput(BaseModel):
    final_summary: str = Field(..., description="Comprehensive summary combining all agent results")
    key_insights: List[str] = Field(..., description="Key insights extracted from all results")
    recommendations: List[str] = Field(default=[], description="Actionable recommendations based on all findings")
    confidence_score: float = Field(..., description="Overall confidence in the aggregated results (0.0 to 1.0)")
    sources_used: List[str] = Field(default=[], description="List of agent sources that contributed to final result")


class AggregationAgent:
    def __init__(self):
        self.llm = LLMGemini()
        self.prompt = BuildContext().context_aggregation("prompts/tasks/aggregation.yml")
        self.model_with_structure = self.llm.model.with_structured_output(AggregationAgentOutput)
        self.agent_type = "aggregation"

    async def run(self, input_data: str, task_context: Dict = None):
        logger.info("Running aggregation agent")
        
        try:
            # Parse input data (should be string representation of task results)
            import ast
            try:
                task_results = ast.literal_eval(input_data)
            except:
                # Fallback: treat as raw string
                task_results = [{"agent_type": "unknown", "result_data": {"summary": input_data}}]

            # Prepare input data from all task results
            aggregation_input = self._prepare_aggregation_input(task_results)
            
            # Generate aggregated result using LLM with fallback
            try:
                llm_response = self.model_with_structure.invoke(
                    self.prompt.format_messages(
                        input=aggregation_input,
                        context=str(task_context or {}),
                        task_count=str(len(task_results))
                    )
                )
                
                if llm_response is None:
                    raise Exception("Structured output returned None")
                
                response = llm_response.model_dump()
                
            except Exception as structured_error:
                logger.warning(f"Structured output failed: {structured_error}, trying fallback")
                
                # Fallback: Use regular LLM without structured output
                fallback_response = self.llm.model.invoke(
                    self.prompt.format_messages(
                        input=aggregation_input,
                        context=str(task_context or {}),
                        task_count=str(len(task_results))
                    )
                )
                
                # Parse response manually
                content = fallback_response.content if hasattr(fallback_response, 'content') else str(fallback_response)
                
                response = {
                    "final_summary": content,
                    "key_insights": self._extract_insights_from_results(task_results),
                    "recommendations": self._extract_recommendations_from_results(task_results),
                    "confidence_score": 0.8,  # Default confidence for fallback
                    "sources_used": [f"{result.get('agent_type', 'unknown')} agent" for result in task_results]
                }
            
            logger.info(f"Aggregation completed with {len(response.get('key_insights', []))} insights")
            return response
            
        except Exception as e:
            logger.error(f"Error running aggregation agent: {e}")
            return {
                "error": str(e),
                "final_summary": f"Aggregation failed: {str(e)}",
                "key_insights": [],
                "recommendations": ["Retry aggregation process", "Check individual agent results"],
                "confidence_score": 0.0,
                "sources_used": []
            }
    
    def _prepare_aggregation_input(self, task_results: List[Dict]) -> str:
        """Prepare structured input for aggregation from all task results"""
        input_parts = ["# Agent Results for Aggregation\n"]
        
        for i, result in enumerate(task_results, 1):
            agent_type = result.get('agent_type', 'unknown')
            status = result.get('status', 'unknown')
            
            input_parts.append(f"## Agent {i}: {agent_type.title()} Agent")
            input_parts.append(f"**Status:** {status}")
            input_parts.append(f"**Execution Time:** {result.get('execution_time', 0):.2f}s")
            
            # Extract result data
            result_data = result.get('result_data', {})
            if result_data and not result_data.get('error'):
                if agent_type == 'summary':
                    summary = result_data.get('summary', 'No summary available')
                    key_points = result_data.get('key_points', [])
                    input_parts.append(f"**Summary:** {summary}")
                    if key_points:
                        input_parts.append("**Key Points:**")
                        for point in key_points:
                            input_parts.append(f"- {point}")
                
                elif agent_type == 'analyst':
                    exec_summary = result_data.get('executive_summary', 'No analysis available')
                    findings = result_data.get('key_findings', [])
                    recommendations = result_data.get('recommendations', [])
                    input_parts.append(f"**Analysis:** {exec_summary}")
                    if findings:
                        input_parts.append("**Key Findings:**")
                        for finding in findings:
                            input_parts.append(f"- {finding}")
                    if recommendations:
                        input_parts.append("**Recommendations:**")
                        for rec in recommendations:
                            input_parts.append(f"- {rec}")
                
                elif agent_type == 'validation':
                    validation_status = result_data.get('overall_status', 'Unknown')
                    validation_summary = result_data.get('validation_summary', 'No validation available')
                    issues = result_data.get('issues_identified', [])
                    input_parts.append(f"**Validation Status:** {validation_status}")
                    input_parts.append(f"**Validation Summary:** {validation_summary}")
                    if issues:
                        input_parts.append("**Issues Identified:**")
                        for issue in issues:
                            input_parts.append(f"- {issue.get('description', 'Unknown issue')}")
            else:
                error_msg = result_data.get('error', 'Unknown error')
                input_parts.append(f"**Error:** {error_msg}")
            
            input_parts.append("")  # Add spacing between agents
        
        return "\n".join(input_parts)
    
    def _extract_insights_from_results(self, task_results: List[Dict]) -> List[str]:
        """Extract key insights from task results for fallback"""
        insights = []
        for result in task_results:
            result_data = result.get('result_data', {})
            agent_type = result.get('agent_type', 'unknown')
            
            if agent_type == 'analyst' and 'key_findings' in result_data:
                insights.extend(result_data['key_findings'])
            elif agent_type == 'summary' and 'key_points' in result_data:
                insights.extend(result_data['key_points'])
        
        return insights[:5]  # Limit to top 5 insights
    
    def _extract_recommendations_from_results(self, task_results: List[Dict]) -> List[str]:
        """Extract recommendations from task results for fallback"""
        recommendations = []
        for result in task_results:
            result_data = result.get('result_data', {})
            if 'recommendations' in result_data:
                recommendations.extend(result_data['recommendations'])
        
        return recommendations[:5]  # Limit to top 5 recommendations
