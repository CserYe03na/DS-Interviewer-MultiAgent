from typing import TypedDict, Optional, List, Dict, Any
import os
from pathlib import Path
from scripts.skill_analyzer_agent import SkillAnalyzerAgent
from scripts.scope_planner_agent import ScopePlannerAgent
from scripts.input_analyzer import TEMPLATE_A
from scripts.skill_mapper import TEMPLATE_B
from scripts.Agent2.retrieval import init_retriever
from scripts.Agent3.Planning_Agent import get_openai_client, normalize_tasks, run_planning_agent
from openai import OpenAI
from langgraph.graph import StateGraph, END

client = OpenAI()
## base_dir = Path(__file__).resolve().parent
## taxonomy_path = base_dir / "data" / "taxonomy_skills.json"
## merge1_path = base_dir / "Agent2" / "merged1.jsonl"



class AgentState(TypedDict):

    # User Input
    jd: str
    user_desc: str
    days_left: int

    # Agent 1 Output
    extracted: Optional[Dict[str, Any]]
    mapped: Optional[Dict[str, Any]]
    weights: Optional[Dict[str, int]]

    # Agent scoop planner Output
    total_questions: Optional[int]  
    difficulty_distribution: Optional[Dict[str, float]]
    plan: Optional[Dict[str, int]]

    # Agent 2 Output
    retrieve_questions: Optional[Dict[str, list]]

    # Agent 3 Output
    days: Optional[List[List[Dict]]] 
    summaries: Optional[List[Dict]]

def agent1(state:AgentState):
    jd = state.get("jd")
    user_desc = state.get("user_desc")
    agent = SkillAnalyzerAgent(client=client)
    result = agent.run(jd_text=jd, user_desc=user_desc)
    state["weights"] = result["weights"]
    return {"extracted":result["extracted"], "mapped":result["mapped"], "weights":result["weights"]}

def agent_scope(state:AgentState):
    jd = state.get("jd")
    user_desc = state.get("user_desc")
    weights = state.get("weights")
    days_left = state.get("days_left") 
    agent = ScopePlannerAgent(client=client)
    result = agent.run(user_desc=user_desc, jd_text=jd, skill_weights=weights, days_left=days_left)
    state["total_questions"] = result["total_questions"]
    state["difficulty_distribution"] = result["difficulty_distribution"]
    state["plan"] = result["skill_plan"]
    return {"total":result["total_questions"], "diff":result["difficulty_distribution"], "plan":result["skill_plan"]}

def agent2(state:AgentState):
    plan = state.get("plan")
    # user_desc = state.get("user_desc")
    retriever = init_retriever()
    retrieve_by_skill = {}
    for skill, k in plan.items():
        num_qs = int(k)
        if num_qs<=0:
            continue
        questions = retriever.retrieve(
            query=skill.replace("_"," "),
            topk=num_qs,
            fetch=200,
            lambda_=0.7,
            type_filter=None,
            skill_filter=skill.replace("_"," "),
        )
        retrieve_by_skill[skill] = questions

    return {"retrieve_questions": retrieve_by_skill}

def agent3(state:AgentState):
    agent2_output = state.get("retrieve_questions")
    days_left = state.get("days_left")
    user_request = f"Create a {days_left} day plan."
    agent2_tasks = []
    for _, task in agent2_output.items():
        agent2_tasks.extend(task)
    agent3_input_tasks = normalize_tasks(agent2_tasks)
    days, summaries = run_planning_agent(
        tasks=agent3_input_tasks,
        user_request=user_request,
        use_llm=True,
        client=client
        )

    return {"days": days, "summaries": summaries}

workflow = StateGraph(AgentState)
workflow.add_node("agent1", agent1)
workflow.add_node("agent_scope", agent_scope)
workflow.add_node("agent2", agent2)
workflow.add_node("agent3", agent3)

workflow.set_entry_point("agent1")
workflow.add_edge("agent1", "agent_scope") 
workflow.add_edge("agent_scope", "agent2")
workflow.add_edge("agent2", "agent3")
workflow.add_edge("agent3", END)

agent_all = workflow.compile()

def multi_agent(jd,user_desc,days_left):
    initial_state : AgentState = {
        "jd": jd,
        "user_desc": user_desc,
        "days_left": days_left
    }
    final_state = agent_all.invoke(initial_state)
    return {
        "days": final_state.get("days"),
        "summaries": final_state.get("summaries"),
        "total_questions": final_state.get("total_questions"),
        "difficulty_distribution": final_state.get("difficulty_distribution"),
        "skill_plan": final_state.get("plan"),
    }


