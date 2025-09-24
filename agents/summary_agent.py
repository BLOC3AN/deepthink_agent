from llms.gemini_llm import LLMGemini
from prompts.build_context import BuildContext
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from mcp_server.mock_tools import MCPToolsManager
from utils.logger import Logger

logger = Logger(__name__)


class SummaryAgentOutput(BaseModel):
    summary: str = Field(..., description="Summary of the document")
    key_points: List[str] = Field(default=[], description="Key points from the content")
    tools_used: List[str] = Field(default=[], description="MCP tools used in summarization")


class SummaryAgent:
    def __init__(self):
        self.llm = LLMGemini()
        self.prompt = BuildContext().context_summary("prompts/tasks/summary.yml")
        self.model_with_structure = self.llm.model.with_structured_output(SummaryAgentOutput)
        self.mcp_tools = MCPToolsManager()
        self.agent_type = "summary"

    async def run(self, input_data: str, task_context: Dict = None):
        logger.info("Running summary agent")

        try:
            # Use MCP tools for enhanced summarization if needed
            tools_used = []
            enhanced_input = input_data

            # Use RAG for additional context
            rag_result = await self.mcp_tools.execute_tool(
                "rag", "retrieve_and_generate",
                query=f"summarization context {input_data}",
                top_k=30
            )
            enhanced_input += f"\n\nAdditional Context: {rag_result}"
            tools_used.append("rag")

            # Generate summary using LLM with fallback
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

            except Exception as structured_error:
                logger.warning(f"Structured output failed: {structured_error}, trying fallback")

                # Fallback: Use regular LLM without structured output
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
                    "summary": content[:500] + "..." if len(content) > 500 else content,
                    "key_points": [
                        "Summary generated using fallback method",
                        "Content extracted from LLM response"
                    ],
                    "tools_used": tools_used
                }

            # Ensure tools_used is in response
            if "tools_used" not in response:
                response["tools_used"] = tools_used

            logger.info("Summary completed successfully!")
            return response

        except Exception as e:
            logger.error(f"Error running summary agent: {e}")
            return {
                "error": str(e),
                "summary": f"Summarization failed: {str(e)}",
                "key_points": [],
                "tools_used": []
            }

