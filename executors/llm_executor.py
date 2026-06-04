from recursive.executor.actions.register import executor_register
from recursive.executor.actions import ActionExecutor
from recursive.llm.llm import OpenAIApiProxy

EXECUTOR_CONFIGS = {
    "chapter_task": {
        "system_prompt": "You are an expert story writer.",
        "user_prompt": "Context:\n{context}\n\nPlease write a comprehensive and compelling long-form narrative section based ONLY on this chapter prompt: {goal}\n\nFormat your output cleanly. Ensure it is a few paragraphs long."
    },
    "report_section_task": {
        "system_prompt": "You are an analytical reporter.",
        "user_prompt": "Context from previous sections:\n{context}\n\nPlease write a comprehensive section for our report based ONLY on this goal/prompt: {goal}\n\nMaintain the original wording and nuance from any source material mentioned. Format cleanly using markdown."
    }
}

@executor_register.register_module()
class LLMExecutor(ActionExecutor):
    def __init__(self):
        super().__init__(actions=[])

    def execute(self, task_node, memory, *args, **kwargs):
        proxy = OpenAIApiProxy()
        goal = task_node.task_info.get("goal", "Do a generic task")
        task_type = task_node.task_info.get("task_type", "generic_task")
        
        # Get model from root node or current task info
        root_node = task_node.node_graph_info.get("root_node")
        model = "gpt-4o"
        if root_node and "model" in root_node.task_info:
            model = root_node.task_info["model"]
        elif "model" in task_node.task_info:
            model = task_node.task_info["model"]
        
        # Use upper graph memory context if available
        context = ""
        if task_node.node_graph_info["layer"] > 0:
            run_info = memory.collect_node_run_info(task_node)
            same_graph = run_info.get("same_graph_precedents", [])
            if same_graph:
                context = "Previous context:\n"
                for precedent in same_graph:
                    context += f"- {precedent.get('result', '')}\n\n"
        
        print(f"\n[Real Agent] Executing Task ({task_type}): {goal} using model {model}")
        
        config = EXECUTOR_CONFIGS.get(task_type)
        if not config:
            print(f"[Real Agent] Warning: No config found for task_type '{task_type}'.")
            return {"result": f"Executed without specific config: {goal}"}

        messages = [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": config["user_prompt"].format(context=context, goal=goal)}
        ]
        
        response = proxy.call(model=model, messages=messages, temperature=0.7, no_cache=True, use_official="azure")
        result_text = response[0]["message"]["content"]
        
        # Save to memory (Aggregate into the main article)
        memory.article += result_text + "\n\n"
        
        print(f"[Real Agent] Finished execution. Result contains {len(result_text.split())} words.\n")
        return {"result": result_text}