from dotenv import load_dotenv
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, MessageGraph, START, END
from utils.nodes import call_model, should_continue, tool_node
from utils.state import AgentState
from helper_functions import save_graph_image

from concurrent.futures import ThreadPoolExecutor
import json

load_dotenv()

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
    Accepts raw user input and preprocesses it into a structured string resume.
    Dynamically detects and processes common resume fields.
    """

    resume_order = [
        "name", "job_title", "summary", "experience", "education",
        "skills", "projects", "achievements", "certifications",
        "tools", "soft_skills", "languages"
    ]

    lines = []

    for field in resume_order:
        if field in raw_input and raw_input[field]:
            value = raw_input[field]

            # ---- Handle nested list-of-dicts fields ----
            if field == "projects" and isinstance(value, list):
                lines.append("Projects:")
                for proj in value:
                    proj_name = proj.get("name", "").strip()
                    impact = proj.get("impact", "").strip()
                    lines.append(f"- {proj_name}: {impact}")

            elif field == "achievements" and isinstance(value, list):
                lines.append("Achievements:")
                for ach in value:
                    lines.append(f"- {ach.strip()}")

            elif isinstance(value, list):
                capitalized_field = field.replace("_", " ").title()
                lines.append(f"{capitalized_field}: {', '.join(item.strip() for item in value)}")

            elif isinstance(value, str):
                capitalized_field = field.replace("_", " ").title()
                lines.append(f"{capitalized_field}: {value.strip()}")

    resume_str = "\n".join(lines)

    return {
        "input_resume": resume_str,
        "prompt_a": "",
        "prompt_b": ""
    }



# ---------- Prompt Builder ----------

# A Agent: Concise, metrics-driven
def build_prompt_a(state: SharedState) -> SharedState:
    input_text = state["input_resume"]

    resume_prompt_a = (
        f'''
        title = Transform a career profile into a concise, metrics-focused professional resume.

        ## Your Role
        You are a Resume Optimization Strategist, specialized in crafting high-impact, scannable resumes for competitive roles. You excel in condensing complex experience into structured, results-oriented bullet points that emphasize measurable outcomes.

        ## Your Task
        <Task_Topic>: Resume Condensation and Results-Driven Structuring
        <Task_Command>: Based on the structured input resume data provided below, reformat the content into a concise professional resume that highlights achievements using metrics, action verbs, and clarity-focused formatting.

        ## Output Application Scenario
        <Background_Context>: The target audience is recruiters and hiring managers who scan resumes quickly and prioritize measurable impact and clarity.
        <Primary_Mission>: Output a tightly formatted resume that maximizes the communication of value and impact within minimal space.
        <Core_Objectives>: Use bullet points, eliminate filler content, apply quantifiable metrics, and preserve all key role outcomes.

        ## Operation Workflow
        <Requirement>: Follow each workflow stage in order
        <Execution Logic>:
        1. <Entry Action>: Analyze the provided resume data to identify core accomplishments and job-critical metrics.
        2. <Transit Action>: Remove filler language, vague terminology, or subjective framing.
        3. <Exit Action>: Reconstruct the resume using compact, bullet-based formatting with focus on tangible results and quantifiable achievements.

        ## Output Rules
        1. Resume must be structured using bullet points in all experience sections.
        2. Prioritize KPIs, percentages, scale, volume, and results.
        3. Use professional tone, avoid narrative, and maintain directness.
        4. Keep formatting clean, using sections: Summary, Experience, Education, Skills.
        5. Avoid storytelling elements or role evolution explanations.

        ## Input Resume:
        {input_text}
        '''
    )
    return {**state, "prompt_a": resume_prompt_a}

# B Agent: Storytelling style
def build_prompt_b(state: SharedState) -> SharedState:
    input_text = state["input_resume"]
    
    resume_prompt_b = (
        f'''
        title = Convert a structured career profile into a storytelling-style resume with humanized narrative and strategic framing.

        ## Your Role
        You are a Narrative Resume Designer, experienced in transforming professional achievements into coherent, compelling personal stories. You specialize in embedding measurable impact into flowing prose that reflects career progression and personal growth.

        ## Your Task
        <Task_Topic>: Story-Centered Resume Generation from Structured Data
        <Task_Command>: Using the structured resume data provided, generate a professional resume in a narrative format. Embed quantifiable accomplishments into clear, human-centered storytelling. Maintain formality but enhance the readability and depth.

        ## Output Application Scenario
        <Background_Context>: The resume will be used to showcase strategic thinking, adaptability, and depth of experience to hiring teams seeking leadership or vision-driven candidates.
        <Primary_Mission>: Balance personal voice and storytelling with professional substance and measurable accomplishments.
        <Core_Objectives>: Present a logical and engaging career narrative while clearly communicating value and outcomes.

        ## Operation Workflow
        <Requirement>: Follow this workflow in order
        <Execution Logic>:
        1. <Entry Action>: Analyze resume content to extract major themes (e.g., growth, innovation, leadership).
        2. <Transit Action>: Integrate those themes into narrative statements using clear prose with embedded performance data.
        3. <Exit Action>: Output a fully structured resume with a storytelling voice across all sections, preserving clarity and professionalism.

        ## Output Rules
        1. Use sentence-based formatting over bullet points (except in skills or tools sections).
        2. Embed quantitative achievements within narrative (e.g., 'led a product launch that generated $1.2M in ARR').
        3. Allow light use of first-person phrasing without pronouns (e.g., 'Focused on optimizing...').
        4. Structure resume using sections: Summary, Experience, Education, Skills, Certifications, etc.
        5. Avoid redundant or filler words; be vivid but efficient.

        ## Input Resume:
        {input_text}
        '''
    )
    return {**state, "prompt_b": resume_prompt_b}

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

    builder.add_edge(START, "build_prompts")
    builder.add_edge("build_prompts", "generate_resumes")
    builder.add_node("evaluate_resumes", evaluate_resumes)
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
    print(f"--- Prompt A --- \n{result_state['prompt_a']}\n")
    print(f"--- Resume A --- \n{result_state['resume_a']}\n")

    print("---------- Prompt B Variant ----------")
    print(f"--- Prompt B --- \n{result_state['prompt_b']}\n")
    print(f"--- Resume B --- \n{result_state['resume_b']}\n")

    print("---------- Evaluation Result ----------")
    print(result_state["evaluation_result"])
