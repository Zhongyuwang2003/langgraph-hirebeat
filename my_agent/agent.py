from dotenv import load_dotenv
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, MessageGraph, START, END
from utils.nodes import call_model, should_continue, tool_node
from utils.state import AgentState
from helper_functions import save_graph_image

from concurrent.futures import ThreadPoolExecutor
import json

load_dotenv()


# Shared State
class PromptBuilderState(TypedDict):
    prompt_variant_a: str
    prompt_variant_b: str
    input_resume: str

# A Agent: Concise, metrics-driven
def build_prompt_a(state: PromptBuilderState) -> PromptBuilderState:
    input_text = state["input_resume"]
    prompt = f"Rephrase this resume concisely with a focus on metrics and achievements:\n\n{input_text}"
    return {**state, "prompt_variant_a": prompt}

# B Agent: Storytelling style
def build_prompt_b(state: PromptBuilderState) -> PromptBuilderState:
    input_text = state["input_resume"]
    prompt = f"Rewrite this resume section using a narrative style that tells a compelling career story:\n\n{input_text}"
    return {**state, "prompt_variant_b": prompt}

# Parallel Wrapper
def parallel_prompt_builders(state: PromptBuilderState) -> PromptBuilderState:
    with ThreadPoolExecutor() as executor:
        future_a = executor.submit(build_prompt_a, state)
        future_b = executor.submit(build_prompt_b, state)
        result_a = future_a.result()
        result_b = future_b.result()
    
    return {
        **state,
        "prompt_variant_a": result_a["prompt_variant_a"],
        "prompt_variant_b": result_b["prompt_variant_b"]
    }

if __name__ == "__main__":
    # Build LangGraph 
    builder = StateGraph(PromptBuilderState)

    builder.add_node("build_prompts", parallel_prompt_builders)

    builder.add_edge(START, "build_prompts")
    builder.add_edge("build_prompts", END)

    prompt_graph = builder.compile()
    save_graph_image(prompt_graph)


    # Testing
    json_path = "sample_input.json"
    try:
        with open(json_path, "r") as f:
            resume_data = json.load(f)

        input_resume = resume_data["input_resume"]
    except FileNotFoundError:
        print(f"Error: {json_path} file was not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON - {e}")
        exit(1)

    initial_state = {
        "input_resume": input_resume,
        "prompt_variant_a": "",
        "prompt_variant_b": ""
    }

    result_state = prompt_graph.invoke(initial_state)

    print("prompt A variant: \n", result_state["prompt_variant_a"])
    print("prompt B variant: \n", result_state["prompt_variant_b"])
