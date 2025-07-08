from typing import TypedDict, Literal

from langgraph.graph import StateGraph, MessageGraph, START, END
from utils.nodes import call_model, should_continue, tool_node
from utils.state import AgentState
from helper_functions import save_graph_image

from concurrent.futures import ThreadPoolExecutor
import json
import os

# ---------- Tracing ----------
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true" 
os.environ["LANGCHAIN_PROJECT"] = "resume-agent" 


# ---------- Shared State ----------
class SharedState(TypedDict):
    input_resume: str
    prompt_a: str
    prompt_b: str
    resume_a: str
    resume_b: str
    evaluation_result: dict


# ---------- Input Intake ----------

# Input Intake Agent: Validates and preprocesses user input
def input_intake_agent(raw_input: dict) -> dict:
    """
    Accepts raw user input and validates required fields:
    - name
    - experience
    - job_title
    - skills

    Returns a cleaned and preprocessed version of the input.
    """
    required_fields = ["name", "experience", "job_title", "skills"]

    # Step 1: Validation
    for field in required_fields:
        if field not in raw_input or not raw_input[field]:
            raise ValueError(f"Missing required field: {field}")

    # Step 2: Preprocessing
    cleaned = {
        "name": raw_input["name"].strip(),
        "experience": raw_input["experience"].strip(),
        "job_title": raw_input["job_title"].strip().title(),
        "skills": [skill.strip().lower() for skill in raw_input["skills"]],
    }

    # Combine fields into a resume string for prompting
    resume_str = f"{cleaned['name']}\n{cleaned['job_title']}\n{cleaned['experience']}\nSkills: {', '.join(cleaned['skills'])}"
    return {
        "input_resume": resume_str,
        "prompt_a": "",
        "prompt_b": ""
    }



# ---------- Prompt Builder ----------

# A Agent: Concise, metrics-driven
def build_prompt_a(state: SharedState) -> SharedState:
    input_text = state["input_resume"]
    prompt = f"Rephrase this resume concisely with a focus on metrics and achievements:\n\n{input_text}"
    return {**state, "prompt_a": prompt}

# B Agent: Storytelling style
def build_prompt_b(state: SharedState) -> SharedState:
    input_text = state["input_resume"]
    prompt = f"Rewrite this resume section using a narrative style that tells a compelling career story:\n\n{input_text}"
    return {**state, "prompt_b": prompt}

# Parallel prompt builder
def parallel_prompt_builders(state: SharedState) -> SharedState:
    with ThreadPoolExecutor() as executor:
        future_a = executor.submit(build_prompt_a, state)
        future_b = executor.submit(build_prompt_b, state)
        result_a = future_a.result()
        result_b = future_b.result()
    
    return {
        **state,
        "prompt_a": result_a["prompt_a"],
        "prompt_b": result_b["prompt_b"]
    }


# ---------- LLM Generator ----------

# Generator function for A/B prompts
def generate_resume_from_prompt(prompt: str, config: dict) -> str:
    state = {"messages": [{"role": "user", "content": prompt}]}
    result = call_model(state, config)
    return result["messages"][0].content

# Parallel resume generator
def llm_generator_agent(state: SharedState) -> SharedState:
    config = {"configurable": {"model_name": "openai"}} 

    with ThreadPoolExecutor() as executor:
        future_a = executor.submit(generate_resume_from_prompt, state["prompt_a"], config)
        future_b = executor.submit(generate_resume_from_prompt, state["prompt_b"], config)
        resume_a = future_a.result()
        resume_b = future_b.result()

    return {
        **state,
        "resume_a": resume_a,
        "resume_b": resume_b
    }


# ---------- Evaluation Agent ----------

def evaluate_resumes(state: SharedState) -> SharedState:
    resume_a = state["resume_a"]
    resume_b = state["resume_b"]

    # 构建评分 prompt
    evaluation_prompt = f"""
        You are a professional resume reviewer. Please evaluate the following two resumes and respond in JSON format like:
        {{
        "a_score": <score out of 10>,
        "b_score": <score out of 10>,
        "winner": "A" or "B",
        "rationale": "short explanation"
        }}

        Resume A:
        {resume_a}

        Resume B:
        {resume_b}
    """

    # 调用 OpenAI 模型
    config = {"configurable": {"model_name": "openai"}}
    state_for_eval = {"messages": [{"role": "user", "content": evaluation_prompt}]}
    result = call_model(state_for_eval, config)
    content = result["messages"][0].content

    try:
        import re
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            json_str = match.group(0)
            evaluation = json.loads(json_str)
        else:
            raise json.JSONDecodeError("No JSON found", content, 0)
    except json.JSONDecodeError:
        evaluation = {
            "a_score": None,
            "b_score": None,
            "winner": "Unknown",
            "rationale": "Failed to parse model output"
        }

    return {
        **state,
        "evaluation_result": evaluation
    }


if __name__ == "__main__":
    # Build LangGraph 
    builder = StateGraph(SharedState)

    builder.add_node("build_prompts", parallel_prompt_builders)
    builder.add_node("generate_resumes", llm_generator_agent)
    builder.add_node("evaluate_resumes", evaluate_resumes)

    builder.add_edge(START, "build_prompts")
    builder.add_edge("build_prompts", "generate_resumes")
    builder.add_edge("generate_resumes", "evaluate_resumes")
    builder.add_edge("evaluate_resumes", END)

    prompt_graph = builder.compile()
    save_graph_image(prompt_graph)


    # Testing
    json_path = "sample_input.json"
    try:
        with open(json_path, "r") as f:
            resume_data = json.load(f)

        initial_state = input_intake_agent(resume_data)
    except FileNotFoundError:
        print(f"Error: {json_path} file was not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON - {e}")
        exit(1)

    result_state = prompt_graph.invoke(initial_state)

    print("---------- Prompt A Variant ----------")
    print(f"*Prompt A*\n{result_state['prompt_a']}\n")
    print(f"*Resume A*\n{result_state['resume_a']}\n")

    print("---------- Prompt B Variant ----------")
    print(f"*Prompt B*\n{result_state['prompt_b']}\n")
    print(f"*Resume B*\n{result_state['resume_b']}\n")

    print("---------- Evaluation Result ----------")
    print(result_state["evaluation_result"])
