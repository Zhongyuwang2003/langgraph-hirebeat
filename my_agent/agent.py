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
    Dynamically handles all provided fields, even those outside a predefined set.
    """
    resume_order = [
        "name", "job_title", "summary", "experience", "education",
        "skills", "projects", "achievements", "certifications",
        "tools", "soft_skills", "languages"
    ]

    lines = []

    # First process known fields in preferred order
    for field in resume_order:
        if field in raw_input and raw_input[field]:
            value = raw_input[field]
            title = field.replace("_", " ").title()

            if isinstance(value, list):
                lines.append(f"{title}:")
                for item in value:
                    if isinstance(item, dict):
                        # Smart formatting for known structures like projects
                        if "name" in item and "impact" in item:
                            lines.append(f"- {item['name']}: {item['impact']}")
                        elif "description" in item:
                            lines.append(f"- {item['description']}")
                        else:
                            combined = "; ".join(f"{k}: {v}" for k, v in item.items())
                            lines.append(f"- {combined}")
                    else:
                        lines.append(f"- {item.strip()}")
            elif isinstance(value, str):
                lines.append(f"{title}: {value.strip()}")

    # Then handle any extra fields not in resume_order
    for field, value in raw_input.items():
        if field not in resume_order and value:
            title = field.replace("_", " ").title()

            if isinstance(value, list):
                lines.append(f"{title}:")
                for item in value:
                    if isinstance(item, dict):
                        combined = "; ".join(f"{k}: {v}" for k, v in item.items())
                        lines.append(f"- {combined}")
                    else:
                        lines.append(f"- {item.strip()}")
            elif isinstance(value, str):
                lines.append(f"{title}: {value.strip()}")

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
    prompt = (
        "You are a resume optimization expert specializing in high-impact, results-driven content.\n\n"
        "## Your Role:\n"
        "Transform the following resume into a concise, metrics-focused version that hiring managers can quickly scan.\n\n"
        "## Requirements:\n"
        "- Use bullet points for clarity.\n"
        "- Focus on quantitative metrics (e.g., % improvements, revenue, users, latency).\n"
        "- Emphasize outcomes and impact using strong action verbs.\n"
        "- Avoid vague or generic descriptions (e.g., 'helped', 'worked on').\n\n"
        f"## Resume:\n{input_text}"
    )
    return {**state, "prompt_a": prompt}

# B Agent: Storytelling style
def build_prompt_b(state: SharedState) -> SharedState:
    input_text = state["input_resume"]
    prompt = (
        "You are a professional resume storytelling expert.\n\n"
        "## Your Role:\n"
        "Rewrite the following resume as a narrative that highlights the candidate’s career journey, growth, and personal contributions.\n\n"
        "## Requirements:\n"
        "- Write in full paragraphs, avoid bullet points.\n"
        "- Emphasize motivation, career progression, and personal development.\n"
        "- Include emotional or team-based elements like mentorship, collaboration, and learning.\n"
        "- Ensure smooth transitions and coherence throughout.\n"
        "- Avoid repeating any sentence or phrase.\n\n"
        f"## Resume:\n{input_text}"
    )
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
    input_resume = state["input_resume"]

    # 根据 input_resume 提取职位类型
    job_title_line = next((line for line in input_resume.splitlines() if "Job Title:" in line), "")
    job_title = job_title_line.replace("Job Title:", "").strip().lower()

    # 判断是否偏向技术岗位
    tech_keywords = ["developer", "engineer", "data", "software", "backend", "frontend", "machine learning"]
    is_technical = any(keyword in job_title for keyword in tech_keywords)

    # 构建 prompt 内容
    if is_technical:
        evaluation_prompt = f"""
You are a hiring manager for a **{job_title_line}** role.

You are reviewing two resumes with different formats:
- **Resume A** uses bullet points and focuses on quantitative outcomes.
- **Resume B** uses storytelling to express the candidate’s career journey.

This role values **technical skill, quantifiable results, impact, and clarity**.

Evaluate both resumes based on how well they:
1. Show technical accomplishments or understanding
2. Communicate measurable results (e.g., latency, revenue, throughput)
3. Highlight initiative, problem-solving, and efficiency
4. Are easy to scan and compare

Use this **5-point rubric per resume**:
- 5: Excellent fit for the technical, metrics-driven nature of the job
- 4: Strong fit with minor gaps
- 3: Somewhat aligned but not compelling
- 2: Weak fit
- 1: Poor fit

Return JSON:
{{
  "a_score": <1-5>,
  "b_score": <1-5>,
  "winner": "A" or "B",
  "rationale": "Why one resume is more effective given the technical nature of the role"
}}

Resume A:
{resume_a}

Resume B:
{resume_b}
"""
    else:
        evaluation_prompt = f"""
You are a hiring manager for a **{job_title_line}** role at a human-centered organization.

You are reviewing two resumes with different formats:
- **Resume A** uses bullet points and focuses on quantitative outcomes.
- **Resume B** uses storytelling to express the candidate’s career journey and motivations.

This role values **empathy, communication, curiosity, and team collaboration** — not just metrics.

Evaluate both resumes based on how well they:
1. Show alignment with the user-centered, soft-skills-heavy nature of the role
2. Convey empathy, motivation, and curiosity
3. Highlight collaboration and personal growth
4. Tell a coherent and persuasive story

Do **not** penalize Resume B for lack of numbers — evaluate the story's fit.

Use this **5-point rubric per resume**:
- 5: Excellent fit for the people-focused nature of the job
- 4: Strong fit with minor gaps
- 3: Somewhat aligned but not compelling
- 2: Weak fit
- 1: Poor fit

Return JSON:
{{
  "a_score": <1-5>,
  "b_score": <1-5>,
  "winner": "A" or "B",
  "rationale": "Why one resume better communicates fit for this empathetic, user-focused role"
}}

Resume A:
{resume_a}

Resume B:
{resume_b}
"""

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
        with open(json_path, "r", encoding="utf-8") as f:
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
