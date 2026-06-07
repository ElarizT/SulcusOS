from kernel.process import AgentProcess

class Parent(AgentProcess):
    name = "Parent"
    restart_backoff_seconds = 0.3
