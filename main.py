import os
import sys

# Ensure the writehere-planner directory is in the path to import 'recursive'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'writehere-planner')))

from recursive.llm.llm import OpenAIApiProxy
from dotenv import load_dotenv

def test_azure_openai():
    # Load environment variables (from .env or api_key.env)
    load_dotenv()
    
    print("Testing Azure OpenAI Integration...")
    
    # Initialize the proxy
    proxy = OpenAIApiProxy(verbose=True)
    
    # Check if Azure environment variables are set
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("Warning: AZURE_OPENAI_ENDPOINT is not set in the environment.")
        print("Please set your Azure OpenAI credentials in .env to run an actual test.")
        return
        
    print(f"Azure Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    
    try:
        # Test 1: Completions (gpt-4o or similar)
        print("\n--- Testing Chat Completions ---")
        model = "gpt-4o"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Azure OpenAI connection successful!'"}
        ]
        
        completion_response = proxy.call(model=model, messages=messages, temperature=0.7, no_cache=True, use_official="azure")
        if completion_response:
            print("Chat Completion Response:")
            print(completion_response[0]["message"]["content"])
        else:
            print("No completion response received.")
            
        # Test 2: Embeddings
        print("\n--- Testing Embeddings ---")
        embedding_model = "text-embedding-3-large"
        text = "Test embedding generation with Azure OpenAI"
        
        embedding_response = proxy.call_embedding(model=embedding_model, text=text, use_official="azure")
        if embedding_response and "data" in embedding_response:
            print(f"Embedding generated successfully. Dimensionality: {len(embedding_response['data'][0]['embedding'])}")
        else:
            print("Failed to generate embeddings.")
            
    except Exception as e:
        print(f"\nTest failed with an error: {e}")

def test_planner_example():
    """
    An example demonstrating how to initialize and run the generic planner engine.
    """
    print("\n--- Running Planner Engine Example ---")
    
    from recursive.graph import TaskStatus, RegularDummyNode, NodeType
    from recursive.engine import GraphRunEngine
    from recursive.executor.actions.register import executor_register
    from recursive.executor.actions import ActionExecutor
    import json
    from recursive.agent.agent_base import agent_register, Agent
    from overrides import overrides

    # Define a real planner that uses the LLM
    @agent_register.register_module()
    class LLMPlanner(Agent):
        def forward(self, node, memory, *args, **kwargs) -> dict:
            # We only plan if we are at the top layer (layer 0)
            layer = node.node_graph_info["layer"]
            if layer >= 1:
                return {"original": "", "result": [], "thought": ""}

            proxy = OpenAIApiProxy()
            goal = node.task_info.get("goal", "Write a story")
            
            print(f"\n[Planner Agent] Outlining Task: {goal}")
            
            messages = [
                {"role": "system", "content": "You are a master story outliner. Break down the user's prompt into exactly 3 chronological chapters. You MUST return ONLY a raw JSON array of objects. Do not wrap it in markdown block quotes like ```json ... ```. Do not include any other text. Each object must have three keys: 'id' (integer starting from 1), 'goal' (string: detailed instruction for writing the chapter), 'task_type' (string: strictly use 'chapter_task'). Example format: [{\"id\": 1, \"goal\": \"Chapter 1: ...\", \"task_type\": \"chapter_task\", \"dependency\": []}]"},
                {"role": "user", "content": f"Create a 3-chapter outline for this prompt: {goal}"}
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
                print(f"[Planner Agent] Successfully created {len(plans)} chapters.")
            except json.JSONDecodeError as e:
                print(f"[Planner Agent] JSON Parse Error. Falling back to default outline. Raw Text: {raw_text}")
                plans = [
                    {"id": 1, "goal": "Chapter 1: The Beginning", "task_type": "chapter_task", "dependency": []},
                    {"id": 2, "goal": "Chapter 2: The Middle", "task_type": "chapter_task", "dependency": [1]},
                    {"id": 3, "goal": "Chapter 3: The End", "task_type": "chapter_task", "dependency": [2]}
                ]
            
            return {
                "original": raw_text,
                "result": plans,
                "thought": ""
            }
            
        def parse_result(self, agent_output, *args, **kwargs) -> dict:
            return agent_output

    # Define a real executor that uses the LLM
    @executor_register.register_module()
    class LLMExecutor(ActionExecutor):
        def __init__(self):
            super().__init__(actions=[])

        def execute(self, task_node, memory, *args, **kwargs):
            proxy = OpenAIApiProxy()
            goal = task_node.task_info.get("goal", "Do a generic task")
            
            # Use upper graph memory context if available
            context = ""
            if task_node.node_graph_info["layer"] > 0:
                run_info = memory.collect_node_run_info(task_node)
                same_graph = run_info.get("same_graph_precedents", [])
                if same_graph:
                    context = "Previous chapters:\n"
                    for precedent in same_graph:
                        context += f"- {precedent.get('result', '')}\n\n"
            
            print(f"\n[Real Agent] Executing Task: {goal}")
            
            messages = [
                {"role": "system", "content": "You are an expert story writer."},
                {"role": "user", "content": f"Context:\n{context}\n\nPlease write a comprehensive and compelling long-form narrative section based ONLY on this chapter prompt: {goal}\n\nFormat your output cleanly. Ensure it is a few paragraphs long."}
            ]
            
            response = proxy.call(model="gpt-4o", messages=messages, temperature=0.7, no_cache=True, use_official="azure")
            result_text = response[0]["message"]["content"]
            
            # Save to memory (Aggregate into the main article)
            memory.article += result_text + "\n\n"
            
            print(f"[Real Agent] Finished writing the section. It contains {len(result_text.split())} words.\n")
            return {"result": result_text}

    # 1. Load configurations for the engine from external file
    with open('planner_config.json', 'r') as f:
        config = json.load(f)

    # 2. Define node metadata
    node_graph_info = {
        "outer_node": None,
        "root_node": None,  # Will assign to itself after initialization
        "parent_nodes": [],
        "layer": 0
    }
    
    task_info = {
        "goal": "Write an epic fantasy story about a young farm boy who discovers a dragon egg.",
        "task_type": "story_task"
    }

    # 3. Create root node
    root_node = RegularDummyNode(
        config=config, 
        nid="0", 
        node_graph_info=node_graph_info, 
        task_info=task_info, 
        node_type=NodeType.PLAN_NODE
    )
    # Self-reference for root node
    root_node.node_graph_info["root_node"] = root_node
    root_node.status = TaskStatus.READY

    # 4. Initialize and Run the Engine
    engine = GraphRunEngine(root_node=root_node, memory_format="xml", config=config)
    
    try:
        final_answer = engine.forward_one_step_untill_done(save_folder="story_output")
        print(f"\nPlanner Execution Complete.\nThe story has been saved to the 'story_output' directory.")
    except Exception as e:
        print(f"\nPlanner failed with an error: {e}")

if __name__ == "__main__":
    test_azure_openai()
    test_planner_example()