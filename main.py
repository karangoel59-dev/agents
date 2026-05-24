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

def test_planner_example(task_goal="Write an epic fantasy story about a young farm boy who discovers a dragon egg.", root_task_type="story_task"):
    """
    An example demonstrating how to initialize and run the generic planner engine.
    """
    print(f"\n--- Running Planner Engine Example ({root_task_type}) ---")
    
    from recursive.graph import TaskStatus, RegularDummyNode, NodeType
    from recursive.engine import GraphRunEngine
    import json

    # Import the custom planners and executors so their decorators register them
    import planners.llm_planner
    import executors.llm_executor

    # 1. Load configurations for the engine from external file
    with open('planner_config.json', 'r') as f:
        engine_config = json.load(f)

    # 2. Define node metadata
    node_graph_info = {
        "outer_node": None,
        "root_node": None,  # Will assign to itself after initialization
        "parent_nodes": [],
        "layer": 0
    }
    
    task_info = {
        "goal": task_goal,
        "task_type": root_task_type
    }

    # 3. Create root node
    root_node = RegularDummyNode(
        config=engine_config, 
        nid="0", 
        node_graph_info=node_graph_info, 
        task_info=task_info, 
        node_type=NodeType.PLAN_NODE
    )
    # Self-reference for root node
    root_node.node_graph_info["root_node"] = root_node
    root_node.status = TaskStatus.READY

    # 4. Initialize and Run the Engine
    engine = GraphRunEngine(root_node=root_node, memory_format="xml", config=engine_config)
    
    try:
        save_folder = f"{root_task_type}_output"
        os.makedirs(save_folder, exist_ok=True)
        final_answer = engine.forward_one_step_untill_done(save_folder=save_folder)
        print(f"\nPlanner Execution Complete.\nThe results have been saved to the '{save_folder}' directory.")
    except Exception as e:
        print(f"\nPlanner failed with an error: {e}")

def create_report_from_md(md_file_path):
    print(f"Reading markdown file from: {md_file_path}")
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File not found: {md_file_path}")
        return
        
    print("Starting report generation using 'report_task'...")
    test_planner_example(task_goal=content, root_task_type="report_task")

if __name__ == "__main__":
    # test_azure_openai()
    md_path = "May 24, 2026 09-36-35 PM Markdown Content.md"
    create_report_from_md(md_path)