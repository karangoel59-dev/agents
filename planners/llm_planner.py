import json
from recursive.agent.agent_base import agent_register, Agent
from recursive.llm.llm import OpenAIApiProxy

PLANNER_CONFIGS = {
    "story_task": {
        "system_prompt": "You are a master story outliner. Break down the user's prompt into exactly 3 chronological chapters. You MUST return ONLY a raw JSON array of objects. Do not wrap it in markdown block quotes like ```json ... ```. Do not include any other text. Each object must have three keys: 'id' (integer starting from 1), 'goal' (string: detailed instruction for writing the chapter), 'task_type' (string: strictly use '{child_task_type}'). Example format: [{{\"id\": 1, \"goal\": \"Chapter 1: ...\", \"task_type\": \"{child_task_type}\", \"dependency\": []}}]",
        "user_prompt": "Create a 3-chapter outline for this prompt: {goal}",
        "child_task_type": "chapter_task",
        "default_plans": [
            {"id": 1, "goal": "Chapter 1: The Beginning", "task_type": "chapter_task", "dependency": []},
            {"id": 2, "goal": "Chapter 2: The Middle", "task_type": "chapter_task", "dependency": [1]},
            {"id": 3, "goal": "Chapter 3: The End", "task_type": "chapter_task", "dependency": [2]}
        ]
    }
    # Add new planner task types here (e.g., 'research_task', 'coding_task')
}

@agent_register.register_module()
class LLMPlanner(Agent):
    def forward(self, node, memory, *args, **kwargs) -> dict:
        # We only plan if we are at the top layer (layer 0)
        layer = node.node_graph_info["layer"]
        if layer >= 1:
            return {"original": "", "result": [], "thought": ""}

        proxy = OpenAIApiProxy()
        goal = node.task_info.get("goal", "Generic goal")
        task_type = node.task_info.get("task_type", "generic_task")
        
        print(f"\n[Planner Agent] Outlining Task ({task_type}): {goal}")
        
        config = PLANNER_CONFIGS.get(task_type)
        if not config:
            print(f"[Planner Agent] Warning: No config found for task_type '{task_type}'. Skipping planning.")
            return {"original": "", "result": [], "thought": ""}
        
        system_prompt = config["system_prompt"].format(child_task_type=config["child_task_type"])
        user_prompt = config["user_prompt"].format(goal=goal)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = proxy.call(model="gpt-4o", messages=messages, temperature=0.7, no_cache=True, use_official="azure")
        raw_text = response[0]["message"]["content"]
        
        # Clean up potential markdown formatting from LLM
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        
        try:
            plans = json.loads(raw_text)
            # Ensure dependencies are set chronologically
            for i, p in enumerate(plans):
                if i == 0:
                    p["dependency"] = []
                else:
                    p["dependency"] = [plans[i-1]["id"]]
            print(f"[Planner Agent] Successfully created {len(plans)} sub-tasks.")
        except json.JSONDecodeError as e:
            print(f"[Planner Agent] JSON Parse Error. Falling back to default outline. Raw Text: {raw_text}")
            plans = config.get("default_plans", [])
        
        return {
            "original": raw_text,
            "result": plans,
            "thought": ""
        }
        
    def parse_result(self, agent_output, *args, **kwargs) -> dict:
        return agent_output