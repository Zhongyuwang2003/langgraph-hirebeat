# Resume AI Agent System


## Multi-Agent System for A/B Testing in Resume Generation Workflow

### Project Overview
We are building a multi-agent system using LangGraph to support A/B testing within a resume generation workflow. The system will enable us to test different strategies (e.g., templates, tones, keyword targeting) by having agents handle distinct sub-tasks and coordinate through a graph-based architecture. This setup allows for dynamic routing, experiment tracking, and improved evaluation of resume effectiveness.

### Objectives 
Design and implement a LangGraph-powered system that: 
- Coordinates multiple agents responsible for various resume generation sub-tasks 
- Supports parallel A/B testing workflows
- Tracks performance and feedback for each version
- Logs all agent interactions for analysis and reproducibility


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
`
