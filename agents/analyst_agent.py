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
    risk_factors: List[str] = Field(default=[], description="Identified risk factors")
    tools_used: List[str] = Field(default=[], description="MCP tools used in analysis")


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
            
            # Generate analysis using LLM
            llm_response = self.model_with_structure.invoke(
                self.prompt.format_messages(
                    input=enhanced_input,
                    context=str(task_context or {}),
                    tools_available=str(tools_used)
                )
            )

            if llm_response is None:
                raise Exception("LLM returned None response")

            response = llm_response.model_dump()

            # Add tools used to response
            response["tools_used"] = tools_used
            
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
