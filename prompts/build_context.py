
from langchain.prompts import ChatPromptTemplate
from prompts.src.load_yml_prompt import load_yml_prompt
import os
from utils.logger import Logger
logger = Logger(__name__)

system_prompt = load_yml_prompt("prompts/tasks/intent_analyst.yml")
prompt = ChatPromptTemplate.from_messages([
    ("system", f"{system_prompt['ROLE']}"),
    ("human", "{input}"),
])

class BuildContext:
    def __init__(self):
        pass

    def context_intent(self, yml_intent_prompt:str):
        try:
            intent_prompt = load_yml_prompt(yml_intent_prompt).get("ROLE")
            logger.info("Loading intent prompt successfully!")
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"{intent_prompt}"),
                ("human", "{input}"),
            ])
            logger.info("Building intent prompt successfully!")
            return prompt
        except Exception as e:
            logger.error(f"Error loading intent prompt: {e}")
            return ""
    
    def context_summary(self, yml_summary_prompt:str):
        try:
            summary_prompt = load_yml_prompt(yml_summary_prompt).get("ROLE")
            logger.info("Loading summary prompt successfully!")
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"{summary_prompt}"),
                ("human", "Input Data: {input}\nTask Context: {context}\nTools Available: {tools_available}"),
            ])
            logger.info("Building summary prompt successfully!")
            return prompt
        except Exception as e:
            logger.error(f"Error loading summary prompt: {e}")
            return ""

    def context_planning(self, yml_planning_prompt: str):
        try:
            planning_prompt = load_yml_prompt(yml_planning_prompt).get("ROLE")
            logger.info("Loading planning prompt successfully!")
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"{planning_prompt}"),
                ("human", "User Input: {input}\nContext: {context}\nAvailable Agents: {agents}"),
            ])
            logger.info("Building planning prompt successfully!")
            return prompt
        except Exception as e:
            logger.error(f"Error loading planning prompt: {e}")
            return ""

    def context_analyst(self, yml_analyst_prompt: str):
        try:
            analyst_prompt = load_yml_prompt(yml_analyst_prompt).get("ROLE")
            logger.info("Loading analyst prompt successfully!")
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"{analyst_prompt}"),
                ("human", "Input Data: {input}\nTask Context: {context}\nTools Available: {tools_available}"),
            ])
            logger.info("Building analyst prompt successfully!")
            return prompt
        except Exception as e:
            logger.error(f"Error loading analyst prompt: {e}")
            return ""

    def context_validation(self, yml_validation_prompt: str):
        try:
            validation_prompt = load_yml_prompt(yml_validation_prompt).get("ROLE")
            logger.info("Loading validation prompt successfully!")
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"{validation_prompt}"),
                ("human", "Input Data: {input}\nTask Context: {context}\nTools Available: {tools_available}"),
            ])
            logger.info("Building validation prompt successfully!")
            return prompt
        except Exception as e:
            logger.error(f"Error loading validation prompt: {e}")
            return ""

    def context_aggregation(self, yml_aggregation_prompt: str):
        try:
            aggregation_prompt = load_yml_prompt(yml_aggregation_prompt).get("ROLE")
            logger.info("Loading aggregation prompt successfully!")
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"{aggregation_prompt}"),
                ("human", "Agent Results: {input}\nTask Context: {context}\nTotal Tasks: {task_count}"),
            ])
            logger.info("Building aggregation prompt successfully!")
            return prompt
        except Exception as e:
            logger.error(f"Error loading aggregation prompt: {e}")
            return ""
