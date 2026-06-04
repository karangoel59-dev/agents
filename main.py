from flask import Flask, request, jsonify, render_template
import os
import datetime
import json
import sys
import threading

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recursive.graph import RegularDummyNode, NodeType, TaskStatus
from recursive.engine import GraphRunEngine

# Import planners and executors to register them with the agent registry
import planners.llm_planner
import executors.llm_executor

app = Flask(__name__)

# --- New additions for async tasks and status polling ---
running_tasks = {}
tasks_lock = threading.Lock()

def get_all_nodes(node):
    """
    Recursively get all nodes in the graph from a starting node.
    """
    nodes = [node]
    for child in node.inner_graph.topological_task_queue:
        nodes.extend(get_all_nodes(child))
    return nodes

def run_task_background(task_id, task_type, goal):
    """
    A helper function to run a task in a background thread.
    """
    output_dir = os.path.join("output", task_id)
    os.makedirs(output_dir, exist_ok=True)

    # Load configurations for the engine from the JSON file
    with open('planner_config.json', 'r') as f:
        config = json.load(f)

    initial_task = {
        "goal": goal,
        "task_type": task_type,
    }

    # Create the root node of the graph
    node_graph_info = {"outer_node": None, "parent_nodes": [], "layer": 0}
    root_node = RegularDummyNode(
        config=config,
        nid="0",
        node_graph_info=node_graph_info,
        task_info=initial_task,
        node_type=NodeType.PLAN_NODE
    )
    root_node.node_graph_info["root_node"] = root_node
    root_node.status = TaskStatus.READY  # Set initial status to READY to start execution

    engine = GraphRunEngine(root_node=root_node, memory_format="xml", config=config)

    try:
        # The engine will handle the execution loop and saving state.
        engine.forward_one_step_untill_done(save_folder=output_dir)
        print(f"\nPlanner Execution Complete for task {task_id}.\nThe results have been saved to the '{output_dir}' directory.")

    except Exception as e:
        print(f"Error running task {task_id}: {e}")
        # Optionally, save error status to a file
        error_info = {"status": "error", "message": str(e)}
        with open(os.path.join(output_dir, "status.json"), "w") as f:
            json.dump(error_info, f)

def find_executing_node(node_data):
    """
    Recursively search for the deepest, non-finished node in the graph,
    which represents the currently executing task.
    """
    if not isinstance(node_data, dict) or node_data.get("status") == "FINISH":
        return None

    # This node is active. Check its children first.
    inner_graph = node_data.get("inner_graph", {})
    if inner_graph and "topological_task_queue" in inner_graph:
        for child_node in inner_graph["topological_task_queue"]:
            active_child = find_executing_node(child_node)
            if active_child:
                return active_child
    
    # If no active child, and this node is not finished, it's the current one.
    return node_data

@app.route('/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """
    API endpoint to poll the status of a task.
    e.g., curl http://127.0.0.1:5000/status/story_task_2026-05-24_22-00-00
    """
    output_dir = os.path.join("output", task_id)
    nodes_file = os.path.join(output_dir, "nodes.json")
    article_file = os.path.join(output_dir, "article.txt")

    response = {"task_id": task_id}

    with tasks_lock:
        thread = running_tasks.get(task_id)

    if not thread and not os.path.exists(output_dir):
        return jsonify({"status": "error", "message": "Task ID not found."}), 404

    is_finished = (thread is None and os.path.exists(article_file)) or \
                  (thread is not None and not thread.is_alive())

    if is_finished:
        response["status"] = "finished"
        if os.path.exists(article_file):
            response["output_file"] = os.path.abspath(article_file)
    elif thread and thread.is_alive():
        response["status"] = "running"
        if os.path.exists(nodes_file):
            with open(nodes_file, 'r') as f:
                graph_data = json.load(f)
            executing_node = find_executing_node(graph_data)
            if executing_node:
                response["current_node"] = {
                    "nid": executing_node.get("nid"),
                    "goal": executing_node.get("task_info", {}).get("goal"),
                    "status": executing_node.get("status")
                }
    else:
        response["status"] = "pending"

    return jsonify(response)

def start_task(task_type, goal):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_id = f"{task_type}_{timestamp}"

    thread = threading.Thread(target=run_task_background, args=(task_id, task_type, goal))
    thread.daemon = True
    thread.start()

    with tasks_lock:
        running_tasks[task_id] = thread

    return {"status": "started", "task_id": task_id}

@app.route('/run/story', methods=['POST'])
def run_story_task():
    data = request.get_json()
    if not data or 'goal' not in data:
        return jsonify({"status": "error", "message": "Missing 'goal' in request body."}), 400
    
    model = data.get('model', 'gpt-4o')
    return jsonify(start_task("story_task", data['goal'], model))

@app.route('/run/report', methods=['POST'])
def run_report_task():
    data = request.get_json()
    if not data or 'goal' not in data:
        return jsonify({"status": "error", "message": "Missing 'goal' in request body."}), 400
    
    model = data.get('model', 'gpt-4o')
    return jsonify(start_task("report_task", data['goal'], model))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/output/<task_id>', methods=['GET'])
def get_task_output(task_id):
    output_dir = os.path.join("output", task_id)
    article_file = os.path.join(output_dir, "article.txt")
    if os.path.exists(article_file):
        with open(article_file, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({"status": "success", "content": content})
    return jsonify({"status": "error", "message": "Output not found"}), 404

@app.route('/history', methods=['GET'])
def get_history():
    output_dir = "output"
    if not os.path.exists(output_dir):
        return jsonify({"status": "success", "history": []})
    
    tasks = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            article_exists = os.path.exists(os.path.join(item_path, "article.txt"))
            tasks.append({
                "task_id": item,
                "finished": article_exists
            })
            
    # Sort by descending timestamp (assuming task_id format ends with timestamp)
    tasks.sort(key=lambda x: x["task_id"], reverse=True)
    return jsonify({"status": "success", "history": tasks})

if __name__ == '__main__':
    app.run(debug=True, port=5000)