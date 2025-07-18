# Resume AI Agent System

### Project Overview
We are building a multi-agent system using LangGraph to support A/B testing within a resume generation workflow. The system will enable us to test different strategies (e.g., templates, tones, keyword targeting) by having agents handle distinct sub-tasks and coordinate through a graph-based architecture. This setup allows for dynamic routing, experiment tracking, and improved evaluation of resume effectiveness.



## LangGraph Workflow (main logic in `agent.py`)

### Shared State
The `SharedState` is a `TypedDict` that defines the shared state among different agents in the system. It includes fields such as `input_resume`, `prompt_a`, `prompt_b`, `resume_a`, `resume_b`, and `evaluation_result`.

### Input Intake Agent
- **Function**: `input_intake_agent`
- **Input**: A dictionary containing raw user input with fields like "name", "job_title", "experience", etc.
- **Output**: A dictionary with the `input_resume` field, which is a pre-processed structured string resume, and empty `prompt_a` and `prompt_b` fields.
- **How it works**: It takes the raw input and processes it according to a predefined order of resume fields. It handles nested list-of-dicts fields like "projects" and "achievements" and formats the output as a string.

### Prompt Builder Agents
- **A Agent**:
  - **Function**: `build_prompt_a`
  - **Input**: A `SharedState` object.
  - **Output**: An updated `SharedState` object with the `prompt_a` field filled. The prompt is designed to generate a concise, metrics-driven resume.
- **B Agent**:
  - **Function**: `build_prompt_b`
  - **Input**: A `SharedState` object.
  - **Output**: An updated `SharedState` object with the `prompt_b` field filled. The prompt is designed to generate a resume in a storytelling style.
- **Parallel Prompt Builder**:
  - **Function**: `parallel_prompt_builders`
  - **Input**: A `SharedState` object.
  - **Output**: An updated `SharedState` object with both `prompt_a` and `prompt_b` fields filled. It uses a `ThreadPoolExecutor` to build the prompts in parallel.

### LLM Generator Agent
- **Function**: `llm_generator_agent`
- **Input**: A `SharedState` object with `prompt_a` and `prompt_b` fields filled.
- **Output**: An updated `SharedState` object with `resume_a` and `resume_b` fields filled. These are the resumes generated by calling the language model with the respective prompts.
- **How it works**: It uses a `ThreadPoolExecutor` to call the `generate_resume_from_prompt` function for both prompts in parallel. The `generate_resume_from_prompt` function calls the language model with the given prompt and returns the generated resume.

### Evaluation Agent
- **Function**: `evaluate_resumes`
- **Input**: A `SharedState` object with `resume_a` and `resume_b` fields filled.
- **Output**: An updated `SharedState` object with the `evaluation_result` field filled. The evaluation result is a dictionary containing scores for both resumes, the winner, and a rationale.
- **How it works**: It builds an evaluation prompt using the two resumes and calls the language model. It then tries to parse the model's output as JSON. If parsing fails, it sets default values in the evaluation result.

### LangGraph Construction
The LangGraph is constructed using the StateGraph class. Nodes are added for each major step in the workflow (`build_prompts`, `generate_resumes`, `evaluate_resumes`), and edges are added to define the flow of the graph. The graph is then compiled and saved as a PNG image using the `save_graph_image` helper function defined in `helper_functions.py`

### Testing
The script loads the sample input from sample_input.json, pre-processes it using the `input_intake_agent`, and then invokes the LangGraph with the initial state. Finally, it prints the generated prompts, resumes, and the evaluation result.



## Setting Up the Conda Environment

Follow the steps below to create and activate a new conda environment, and install all required dependencies.

### 1. Create a New Conda Environment

Use the command below to create a new environment.

```bash
conda create --name langgraph-resume python=3.10
```

### 2. Activate the Environment

```bash
conda activate langgraph-resume
```

### 3. Install Required Packages

Make sure your terminal is in the `my_agent` directory of the project (where `requirements.txt` is located), then run:

```bash
pip install -r requirements.txt
```


## Multi-Agent System for A/B Testing in Resume Generation Workflow

### Objectives 
Design and implement a LangGraph-powered system that: 
- Coordinates multiple agents responsible for various resume generation sub-tasks 
- Supports parallel A/B testing workflows
- Tracks performance and feedback for each version
- Logs all agent interactions for analysis and reproducibility

### System Functions
#### 1. Input Intake Agent
- Accepts raw user data (name, experience, job title, skills) 
- Validates completeness and pre-processes input

#### 2. Prompt Builder Agents (A/B Variants)
- A Agent: Uses a concise, metrics-driven prompt style
- B Agent: Uses a storytelling, narrative-focused style
- Each builds a prompt for the LLM based on the same input

#### 3. LLM Generator Agent
- Takes prompts from A and B branches
- Calls the language model to generate corresponding resumes
- Sends output to evaluation agents

#### 4. Evaluation Agents
- Responsible for scoring the resumes from both A and B branches
- You will need to define and justify appropriate evaluation metrics (e.g., keyword density, readability, role alignment)
- Optionally call external resume ranking APIs
- Store scoring data in experiment logs

#### 5. Logger/Orchestrator Node
- Records agent paths, inputs, and outputs
- Helps visualize graph traversal and performance metrics

#### 6. Experiment Manager (Optional Stretch)
- Dynamically adjusts A/B variants
- Incorporates user feedback loop (thumbs up/down)

