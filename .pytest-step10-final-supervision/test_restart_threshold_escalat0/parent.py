from kernel.process import AgentProcess

class Parent(AgentProcess):
    name = "Parent"
    max_restarts = 1
    restart_window_seconds = 10.0
