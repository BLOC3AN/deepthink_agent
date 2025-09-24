import asyncio
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

from agents.intent_agent import PlanningAgent
from core.task_manager import TaskManager, ExecutionPlan
from database.mongodb_client import MongoDBClient
from database.seed_data import seed_database
from utils.logger import Logger

logger = Logger(__name__)


class DeepModelSystem:
    """Main system orchestrator for the multi-agent workflow"""
    
    def __init__(self):
        self.planning_agent = PlanningAgent()
        self.task_manager = TaskManager()
        self.mongodb = MongoDBClient()
        logger.info("DeepModel System initialized")
    
    async def initialize(self):
        """Initialize the system and seed database if needed"""
        logger.info("Initializing DeepModel System...")
        
        # Connect to MongoDB and seed data
        connected = await self.mongodb.connect()
        if not connected:
            logger.error("Failed to connect to MongoDB")
            return False
        
        # Check if we need to seed data
        agents = await self.mongodb.get_all_agent_cards()
        if len(agents) == 0:
            logger.info("No agent cards found, seeding database...")
            await seed_database(self.mongodb)
        
        await self.mongodb.disconnect()
        logger.info("System initialization completed")
        return True
    
    async def process_request(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """Process a complete user request through the multi-agent system"""
        logger.info(f"Processing request: {user_input[:100]}...")
        
        try:
            # Step 1: Planning phase
            logger.info("🧠 Starting planning phase...")
            planning_result = await self.planning_agent.run(user_input, context)
            
            if "error" in planning_result:
                return {
                    "status": "failed",
                    "error": planning_result["error"],
                    "phase": "planning"
                }
            
            # Step 2: Create execution plan
            execution_plan = ExecutionPlan(
                session_id=planning_result.get("session_id", "unknown"),
                detected_intents=planning_result.get("detected_intents", []),
                execution_strategy=planning_result.get("execution_strategy", "parallel"),
                tasks=planning_result.get("tasks", []),
                estimated_time=planning_result.get("estimated_time", 0)
            )
            
            logger.info(f"📋 Execution plan created with {len(execution_plan.tasks)} tasks")
            
            # Step 3: Execute tasks
            logger.info("🚀 Starting task execution...")
            aggregated_result = await self.task_manager.execute_plan(execution_plan)
            
            # Step 4: Return final result
            return {
                "status": "success",
                "session_id": aggregated_result.session_id,
                "planning_result": {
                    "detected_intents": planning_result.get("detected_intents", []),
                    "execution_strategy": planning_result.get("execution_strategy", ""),
                    "estimated_time": planning_result.get("estimated_time", 0)
                },
                "execution_result": {
                    "overall_status": aggregated_result.overall_status,
                    "execution_summary": aggregated_result.execution_summary,
                    "total_execution_time": aggregated_result.total_execution_time,
                    "task_count": len(aggregated_result.task_results),
                    "task_results": aggregated_result.task_results
                },
                "final_output": aggregated_result.final_output,
                "timestamp": aggregated_result.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "phase": "execution"
            }


async def test_end_to_end():
    """Test the complete end-to-end workflow"""
    logger.info("🧪 Starting End-to-End Integration Test")
    
    # Initialize system
    system = DeepModelSystem()
    initialized = await system.initialize()
    
    if not initialized:
        logger.error("❌ System initialization failed")
        return False
    
    # Test case 1: Complex multi-intent request
    logger.info("📝 Test Case 1: Multi-intent request")
    test_input = "Analyze the market trends for AI technology in 2024, summarize the key findings, and validate the accuracy of the data sources used"
    test_context = {
        "domain": "technology",
        "timeframe": "2024",
        "analysis_type": "market_trends"
    }
    
    result = await system.process_request(test_input, test_context)
    
    if result["status"] == "success":
        logger.info("✅ Test Case 1 PASSED")
        logger.info(f"   - Session ID: {result['session_id']}")
        logger.info(f"   - Detected intents: {result['planning_result']['detected_intents']}")
        logger.info(f"   - Execution strategy: {result['planning_result']['execution_strategy']}")
        logger.info(f"   - Overall status: {result['execution_result']['overall_status']}")
        logger.info(f"   - Total execution time: {result['execution_result']['total_execution_time']:.2f}s")
        logger.info(f"   - Final output length: {len(result['final_output'])} characters")
    else:
        logger.error(f"❌ Test Case 1 FAILED: {result.get('error', 'Unknown error')}")
        return False
    
    # Test case 2: Simple single-intent request
    logger.info("📝 Test Case 2: Single-intent request")
    test_input_2 = "Please summarize this technical document about machine learning algorithms"
    
    result_2 = await system.process_request(test_input_2)
    
    if result_2["status"] == "success":
        logger.info("✅ Test Case 2 PASSED")
        logger.info(f"   - Detected intents: {result_2['planning_result']['detected_intents']}")
        logger.info(f"   - Task count: {result_2['execution_result']['task_count']}")
    else:
        logger.error(f"❌ Test Case 2 FAILED: {result_2.get('error', 'Unknown error')}")
        return False
    
    logger.info("🎉 All End-to-End tests PASSED!")
    return True


async def main():
    """Main application entry point"""
    logger.info("🚀 Starting DeepModel Multi-Agent System")
    
    # Run integration tests
    test_passed = await test_end_to_end()
    
    if test_passed:
        logger.info("✅ System is ready for production use!")
        
        # Example of how to use the system
        system = DeepModelSystem()
        await system.initialize()
        
        # Process a sample request
        sample_request = "Analyze customer feedback data and provide actionable insights"
        result = await system.process_request(sample_request)
        
        print("\n" + "="*50)
        print("SAMPLE REQUEST RESULT:")
        print("="*50)
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Final Output:\n{result['final_output']}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        print("="*50)
        
    else:
        logger.error("❌ System tests failed!")


if __name__ == "__main__":
    asyncio.run(main())
