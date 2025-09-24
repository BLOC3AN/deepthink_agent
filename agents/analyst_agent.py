from llms.gemini_llm import LLMGemini
from prompts.build_context import BuildContext
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from mcp_server.mock_tools import MCPToolsManager
from utils.logger import Logger

logger = Logger(__name__)


class AnalysisInsight(BaseModel):
    finding: str = Field(..., description="Key analytical finding")
    evidence: str = Field(..., description="Supporting evidence for the finding")
    confidence: float = Field(..., description="Confidence level (0.0 to 1.0)")


class AnalystAgentOutput(BaseModel):
    executive_summary: str = Field(..., description="Executive summary of analysis")
    key_findings: List[AnalysisInsight] = Field(..., description="List of key analytical findings")
    detailed_analysis: str = Field(..., description="Detailed analysis with explanations")
    recommendations: List[str] = Field(..., description="Actionable recommendations")
    risk_factors: Optional[List[str]] = Field(default=[], description="Identified risk factors")
    tools_used: Optional[List[str]] = Field(default=[], description="MCP tools used in analysis")


class AnalystAgent:
    def __init__(self):
        self.llm = LLMGemini()
        self.prompt = BuildContext().context_analyst("prompts/tasks/analyst.yml")
        self.model_with_structure = self.llm.model.with_structured_output(AnalystAgentOutput)
        self.mcp_tools = MCPToolsManager()
        self.agent_type = "analyst"

    async def run(self, input_data: str, task_context: Dict = None):
        logger.info("Running analyst agent")
        
        try:
            # Use MCP tools for data gathering if needed
            tools_used = []
            enhanced_input = input_data
            
            # Example: Use web search for market data
            if "market" in input_data.lower() or "trend" in input_data.lower():
                search_result = await self.mcp_tools.execute_tool(
                    "websearch", "search", 
                    query=f"market trends analysis {input_data[:50]}", 
                    limit=5
                )
                enhanced_input += f"\n\nMarket Research: {search_result}"
                tools_used.append("websearch")
            
            # Example: Use RAG for knowledge base
            if "analysis" in input_data.lower() or "data" in input_data.lower():
                rag_result = await self.mcp_tools.execute_tool(
                    "rag", "retrieve_and_generate",
                    query=f"analytical insights {input_data[:50]}",
                    top_k=3
                )
                enhanced_input += f"\n\nKnowledge Base: {rag_result}"
                tools_used.append("rag")
            
            # Example: Use SQL for data queries
            if "performance" in input_data.lower() or "metrics" in input_data.lower():
                sql_result = await self.mcp_tools.execute_tool(
                    "sql", "execute_query",
                    query="SELECT * FROM performance_metrics ORDER BY date DESC LIMIT 10",
                    params={"analysis_type": "performance"}
                )
                enhanced_input += f"\n\nData Query: {sql_result}"
                tools_used.append("sql")
            
            # Generate analysis using LLM with robust error handling
            try:
                llm_response = self.model_with_structure.invoke(
                    self.prompt.format_messages(
                        input=enhanced_input,
                        context=str(task_context or {}),
                        tools_available=str(tools_used)
                    )
                )

                if llm_response is None:
                    raise Exception("Structured output returned None")

                response = llm_response.model_dump()
                logger.info("✅ LLM structured output successful")

            except Exception as structured_error:
                logger.warning(f"LLM structured output failed: {structured_error}, using fallback")

                # Fallback: Use regular LLM without structured output
                try:
                    fallback_response = self.llm.model.invoke(
                        self.prompt.format_messages(
                            input=enhanced_input,
                            context=str(task_context or {}),
                            tools_available=str(tools_used)
                        )
                    )

                    # Parse response manually
                    content = fallback_response.content if hasattr(fallback_response, 'content') else str(fallback_response)

                    response = {
                        "executive_summary": content[:300] + "..." if len(content) > 300 else content,
                        "key_findings": [],
                        "detailed_analysis": content,
                        "recommendations": ["Analysis completed using fallback method"],
                        "risk_factors": [],
                        "tools_used": tools_used
                    }
                    logger.info("✅ Fallback response created successfully")

                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise Exception(f"Both structured and fallback LLM calls failed: {structured_error}")

            # Normalize and validate response
            response = self._normalize_response(response, tools_used)

            logger.info(f"Analysis completed with {len(response.get('key_findings', []))} findings")
            return response
            
        except Exception as e:
            logger.error(f"Error running analyst agent: {e}")
            return {
                "error": str(e),
                "executive_summary": "Analysis failed due to technical error",
                "key_findings": [],
                "detailed_analysis": f"Error occurred: {str(e)}",
                "recommendations": ["Retry analysis with different parameters"],
                "risk_factors": ["Technical failure in analysis process"],
                "tools_used": []
            }

    def _normalize_response(self, response: dict, tools_used: list) -> dict:
        """Normalize and validate response to ensure all fields are properly formatted"""

        # Ensure all required fields exist with proper types
        normalized = {
            "executive_summary": response.get("executive_summary", "Analysis completed"),
            "key_findings": response.get("key_findings", []),
            "detailed_analysis": response.get("detailed_analysis", "No detailed analysis available"),
            "recommendations": response.get("recommendations", []),
            "risk_factors": response.get("risk_factors") or [],  # Handle None case
            "tools_used": response.get("tools_used") or tools_used  # Handle None case
        }

        # Ensure lists are actually lists (handle None values)
        if normalized["key_findings"] is None:
            normalized["key_findings"] = []
        if normalized["recommendations"] is None:
            normalized["recommendations"] = []
        if normalized["risk_factors"] is None:
            normalized["risk_factors"] = []
        if normalized["tools_used"] is None:
            normalized["tools_used"] = tools_used

        # Ensure strings are actually strings
        if not isinstance(normalized["executive_summary"], str):
            normalized["executive_summary"] = str(normalized["executive_summary"])
        if not isinstance(normalized["detailed_analysis"], str):
            normalized["detailed_analysis"] = str(normalized["detailed_analysis"])

        # Always include tools_used from actual execution
        normalized["tools_used"] = tools_used

        logger.info(f"✅ Response normalized: {len(normalized['key_findings'])} findings, {len(normalized['recommendations'])} recommendations")
        return normalized
