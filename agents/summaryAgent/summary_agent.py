from llms.gemini_llm import LLMGemini
from prompts.build_context import BuildContext
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from utils.logger import Logger

logger = Logger(__name__)


class SummaryAgentOutput(BaseModel):
    summary: str = Field(..., description="Summary of the document")


class SummaryAgent:
    def __init__(self):
        self.llm = LLMGemini()
        self.prompt = BuildContext().context_summary("prompts/tasks/summary.yml")
        self.model_with_structure = self.llm.model.with_structured_output(SummaryAgentOutput)
        self.agent_type = "summary"

    def _validate_output(self, response: Dict) -> bool:
        """Validate if the response has the required fields"""
        required_fields = ["summary"]
        return all(field in response for field in required_fields)

    def run(self, input_data: str, task_context: Dict = None):
        logger.info("Running summary agent")
        max_retries = 3
        attempt = 0

        while attempt < max_retries:
            attempt += 1
            logger.info(f"Attempt {attempt}/{max_retries}")

            try:
                # Try structured output first
                try:
                    llm_response = self.model_with_structure.invoke(
                        self.prompt.format_messages(
                            input=input_data,
                            context=str(task_context or {}),
                            tools_available="[]"
                        )
                    )

                    if llm_response is None:
                        raise Exception("Structured output returned None")

                    response = llm_response.model_dump()

                    # Validate output format
                    if self._validate_output(response):
                        logger.info(f"✅ Structured output successful on attempt {attempt}")
                        return response
                    else:
                        logger.warning(f"Invalid output format on attempt {attempt}, retrying...")
                        continue

                except Exception as structured_error:
                    logger.warning(f"Structured output failed on attempt {attempt}: {structured_error}")

                    # Fallback: Use regular LLM without structured output
                    try:
                        fallback_response = self.llm.model.invoke(
                            self.prompt.format_messages(
                                input=input_data,
                                context=str(task_context or {}),
                                tools_available="[]"
                            )
                        )

                        # Parse response manually
                        content = fallback_response.content if hasattr(fallback_response, 'content') else str(fallback_response)

                        response = {"summary": content}

                        # Validate fallback output
                        if self._validate_output(response):
                            logger.info(f"✅ Fallback response successful on attempt {attempt}")
                            return response
                        else:
                            logger.warning(f"Invalid fallback output on attempt {attempt}, retrying...")
                            continue

                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed on attempt {attempt}: {fallback_error}")
                        if attempt == max_retries:
                            raise Exception(f"Both structured and fallback failed after {max_retries} attempts")
                        continue

            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    logger.error(f"All {max_retries} attempts failed")
                    return {"error": str(e)}
                continue

        # If we get here, all attempts failed
        return {"error": f"Failed to generate valid output after {max_retries} attempts"}
