from llms.gemini_llm import LLMGemini
from prompts.build_context import BuildContext
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from mcp_server.mock_tools import MCPToolsManager
from utils.logger import Logger

logger = Logger(__name__)


class ValidationIssue(BaseModel):
    issue_type: str = Field(..., description="Type of validation issue")
    description: str = Field(..., description="Description of the issue")
    severity: str = Field(..., description="Severity level: high, medium, low")
    evidence: str = Field(..., description="Evidence supporting the issue identification")


class ValidationResult(BaseModel):
    item: str = Field(..., description="Item being validated")
    status: str = Field(..., description="Validation status: VALID, INVALID, UNCERTAIN")
    confidence: float = Field(..., description="Confidence level (0.0 to 1.0)")
    reasoning: str = Field(..., description="Reasoning for validation decision")


class ValidationAgentOutput(BaseModel):
    overall_status: str = Field(..., description="Overall validation status")
    validation_summary: str = Field(..., description="Summary of validation process")
    validation_results: List[ValidationResult] = Field(..., description="Detailed validation results")
    issues_identified: List[ValidationIssue] = Field(..., description="Issues found during validation")
    recommendations: List[str] = Field(..., description="Recommendations for improvement")
    sources_checked: Optional[List[str]] = Field(default=[], description="Sources used for validation")
    tools_used: Optional[List[str]] = Field(default=[], description="MCP tools used in validation")


class ValidationAgent:
    def __init__(self):
        self.llm = LLMGemini()
        self.prompt = BuildContext().context_validation("prompts/tasks/validation.yml")
        self.model_with_structure = self.llm.model.with_structured_output(ValidationAgentOutput)
        self.mcp_tools = MCPToolsManager()
        self.agent_type = "validation"

    async def run(self, input_data: str, task_context: Dict = None):
        logger.info("Running validation agent")
        
        try:
            # Use MCP tools for validation checks
            tools_used = []
            enhanced_input = input_data
            
            # Use web search for fact checking
            if any(keyword in input_data.lower() for keyword in ["fact", "claim", "statement", "data"]):
                search_result = await self.mcp_tools.execute_tool(
                    "websearch", "search",
                    query=f"fact check verify {input_data[:50]}",
                    limit=3
                )
                enhanced_input += f"\n\nFact Check Sources: {search_result}"
                tools_used.append("websearch")
            
            # Use RAG for knowledge verification
            if any(keyword in input_data.lower() for keyword in ["information", "knowledge", "reference"]):
                rag_result = await self.mcp_tools.execute_tool(
                    "rag", "retrieve_and_generate",
                    query=f"verify information {input_data[:50]}",
                    top_k=3
                )
                enhanced_input += f"\n\nKnowledge Verification: {rag_result}"
                tools_used.append("rag")
            
            # Use SQL for data validation
            if any(keyword in input_data.lower() for keyword in ["database", "record", "data", "metrics"]):
                sql_result = await self.mcp_tools.execute_tool(
                    "sql", "execute_query",
                    query="SELECT * FROM validation_sources WHERE status = 'active'",
                    params={"validation_type": "data_check"}
                )
                enhanced_input += f"\n\nData Validation: {sql_result}"
                tools_used.append("sql")
            
            # Generate validation using LLM
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
            
            logger.info(f"Validation completed with status: {response.get('overall_status', 'UNKNOWN')}")
            return response
            
        except Exception as e:
            logger.error(f"Error running validation agent: {e}")
            return {
                "error": str(e),
                "overall_status": "ERROR",
                "validation_summary": f"Validation failed due to technical error: {str(e)}",
                "validation_results": [],
                "issues_identified": [{
                    "issue_type": "technical_error",
                    "description": f"Validation process failed: {str(e)}",
                    "severity": "high",
                    "evidence": "System error during validation"
                }],
                "recommendations": ["Retry validation process", "Check system configuration"],
                "sources_checked": [],
                "tools_used": []
            }
