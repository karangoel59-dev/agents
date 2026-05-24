import json
from recursive.agent.agent_base import agent_register, Agent
from recursive.llm.llm import OpenAIApiProxy
from copy import deepcopy

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
    },
    "report_task": {
        "system_prompt": "You are a master report outliner. Break down the user's prompt into a detailed, multi-section report structure with 5-7 chronological sections. You MUST return ONLY a raw JSON array of objects. Do not wrap it in markdown block quotes like ```json ... ```. Do not include any other text. Each object must have three keys: 'id' (integer starting from 1), 'goal' (string: detailed instruction for writing the section), 'task_type' (string: strictly use '{child_task_type}'). The plan should cover introduction, analysis of different aspects, and a conclusion.",
        "user_prompt": "Create a multi-section report outline for this topic: {goal}",
        "child_task_type": "report_section_task",
        "default_plans": [
            {"id": 1, "goal": "Section 1: Introduction", "task_type": "report_section_task", "dependency": []},
            {"id": 2, "goal": "Section 2: Main Body", "task_type": "report_section_task", "dependency": [1]},
            {"id": 3, "goal": "Section 3: Conclusion", "task_type": "report_section_task", "dependency": [2]}
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
            plans = json.loads(raw_text) if raw_text else []

            if task_type in ["report_task", "story_task"]:
                tone_guideline_task = {
                    "goal": "Analyze the original text to define its tone and style. Create a concise guideline for all subsequent writing tasks to follow, ensuring consistency in voice, vocabulary, and sentiment. This is a preliminary step; output only the guidelines.",
                    "task_type": config["child_task_type"],
                }
                plans.insert(0, tone_guideline_task)
                print(f"[Planner Agent] Injected tone guideline task.")

            # Ensure dependencies are set chronologically and IDs are correct
            for i, p in enumerate(plans):
                p["id"] = i + 1
                if i == 0:
                    p["dependency"] = []
                else:
                    p["dependency"] = [i]
            print(f"[Planner Agent] Successfully created {len(plans)} sub-tasks.")
        except json.JSONDecodeError as e:
            print(f"[Planner Agent] JSON Parse Error. Falling back to default outline. Raw Text: {raw_text}")
            plans = deepcopy(config.get("default_plans", []))

            if task_type in ["report_task", "story_task"]:
                tone_guideline_task = {
                    "id": 1,
                    "goal": "Analyze the original text to define its tone and style. Create a concise guideline for all subsequent writing tasks to follow, ensuring consistency in voice, vocabulary, and sentiment. This is a preliminary step; output only the guidelines.",
                    "task_type": config["child_task_type"],
                    "dependency": []
                }
                for p in plans:
                    p["id"] += 1
                    p["dependency"] = [d + 1 for d in p["dependency"]] if p.get("dependency") else [1]
                plans.insert(0, tone_guideline_task)
                print(f"[Planner Agent] Injected tone guideline task into default plan.")
        
        return {
            "original": raw_text,
            "result": plans,
            "thought": ""
        }
        
    def parse_result(self, agent_output, *args, **kwargs) -> dict:
        return agent_output