# WriteHERE Recursive Planner Engine

This is a standalone, generic Recursive DAG Agentic framework extracted from the [WriteHERE](https://github.com/microsoft/WriteHERE) project.

## Overview
The framework executes a nested Directed Acyclic Graph (DAG) of tasks. The engine was originally designed to perform deep recursive planning for report and story generation, but it has been stripped of writing-specific hardcoded logic so it can be applied to **any domain** (e.g. software development, data analysis).

## How to adapt for a new domain:
As described in the extraction guide:

1. **Customize the Planning Prompts:**
   You will need to write custom prompts that instruct the LLM on your domain's specific Task Types (e.g., `code`, `test`, `debug`). Ensure the LLM outputs the standard nested JSON format expected by the engine.

2. **Register Custom Executors:**
   For any new task type introduced (e.g., `code`), create and register a new executor:

   ```python
   from recursive.executor.actions.register import executor_register
   from recursive.executor.actions import ActionExecutor

   @executor_register.register_module("code")
   class CodeExecutor(ActionExecutor):
       def execute(self, task_node, memory):
           # Your custom logic to execute the coding task
           pass
   ```

3. **Provide Config to `GraphRunEngine`:**
   Instantiate the `GraphRunEngine` by providing a configuration dictionary containing your custom mapping from task type to executors, prompt versions, etc.

## Installation

You can install the package locally:
```bash
pip install -e .
```
