from datetime import datetime
from typing import List, Dict


def get_agent_cards_seed_data() -> List[Dict]:
    """Sample agent cards for testing"""
    return [
        {
            "id": "summary-agent-alpha",
            "name": "Summary Agent Alpha",
            "description": "Specialized summary agent for technical documents",
            "type": "summary",
            "status": "active",
            "version": "1.0.0",
            "url": "http://summary-agent-alpha:8080",
            "capabilities": ["technical_summary", "document_analysis"],
            "parameters": {
                "max_input_length": 10000,
                "timeout": 30,
                "specialization": "technical_docs"
            },
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        },
        {
            "id": "summary-agent-beta",
            "name": "Summary Agent Beta",
            "description": "Summary agent for business reports and documents",
            "type": "summary",
            "status": "active",
            "version": "1.0.0",
            "url": "http://summary-agent-beta:8081",
            "capabilities": ["business_summary", "report_analysis"],
            "parameters": {
                "max_input_length": 8000,
                "timeout": 25,
                "specialization": "business_reports"
            },
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        },
        {
            "id": "analyst-agent-gamma",
            "name": "Analyst Agent Gamma",
            "description": "Data analysis and trend identification agent",
            "type": "analyst",
            "status": "active",
            "version": "1.0.0",
            "url": "http://analyst-agent-gamma:8082",
            "capabilities": ["data_analysis", "trend_identification", "statistical_analysis"],
            "parameters": {
                "max_input_length": 15000,
                "timeout": 45,
                "specialization": "data_analysis"
            },
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        },
        {
            "id": "analyst-agent-delta",
            "name": "Analyst Agent Delta",
            "description": "Market research and competitive analysis agent",
            "type": "analyst",
            "status": "active",
            "version": "1.0.0",
            "url": "http://analyst-agent-delta:8083",
            "capabilities": ["market_research", "competitive_analysis", "industry_insights"],
            "parameters": {
                "max_input_length": 12000,
                "timeout": 40,
                "specialization": "market_research"
            },
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        },
        {
            "id": "validation-agent-epsilon",
            "name": "Validation Agent Epsilon",
            "description": "Fact checking and data validation agent",
            "type": "validation",
            "status": "active",
            "version": "1.0.0",
            "url": "http://validation-agent-epsilon:8084",
            "capabilities": ["fact_checking", "data_validation", "consistency_check"],
            "parameters": {
                "max_input_length": 10000,
                "timeout": 35,
                "specialization": "fact_checking"
            },
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
    ]


async def seed_database(mongodb_client):
    """Seed database with initial data"""
    from utils.logger import Logger
    logger = Logger(__name__)
    
    try:
        # Clear existing data
        await mongodb_client.db.agent_cards.delete_many({})
        
        # Insert agent cards
        agent_cards = get_agent_cards_seed_data()
        await mongodb_client.db.agent_cards.insert_many(agent_cards)
        
        logger.info(f"Seeded {len(agent_cards)} agent cards")
        return True
    except Exception as e:
        logger.error(f"Error seeding database: {e}")
        return False


if __name__ == "__main__":
    import asyncio
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from database.mongodb_client import MongoDBClient

    async def main():
        client = MongoDBClient()
        await client.connect()
        success = await seed_database(client)
        await client.disconnect()
        print(f"Seeding {'successful' if success else 'failed'}")

    asyncio.run(main())
