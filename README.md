# Resume AI Agent System

### Project Overview
We are building a multi-agent system using LangGraph to support A/B testing within a resume generation workflow. The system will enable us to test different strategies (e.g., templates, tones, keyword targeting) by having agents handle distinct sub-tasks and coordinate through a graph-based architecture. This setup allows for dynamic routing, experiment tracking, and improved evaluation of resume effectiveness.


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

