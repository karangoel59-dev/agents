from recursive.utils.register import Register
from recursive.agent.agent_base import agent_register
from recursive.executor.actions.register import executor_register

class AgentProxy:
    def __init__(self, config):
        self.config = config
    
    def proxy(self, action_name):
        # We look up the action_mapping in config
        if action_name in self.config.get("action_mapping", {}):
            mapped_name = self.config["action_mapping"][action_name]
            
            # Usually it returns an agent that has a 'forward' method
            # If it's an executor, we can wrap it
            if mapped_name in executor_register.module_dict:
                executor_cls = executor_register.module_dict.get(mapped_name)
                executor = executor_cls() # Initialize with dummy actions if needed
                
                # Wrap it in an object with a forward method
                class ExecutorWrapper:
                    def forward(self, node, memory, *args, **kwargs):
                        return executor.execute(node, memory, *args, **kwargs)
                return ExecutorWrapper()
                
            elif mapped_name in agent_register.module_dict:
                agent_cls = agent_register.module_dict.get(mapped_name)
                return agent_cls()
        
        # Fallback dummy
        class DummyAgent:
            def forward(self, node, memory, *args, **kwargs):
                return {"result": f"Dummy result for {action_name}"}
        return DummyAgent()
