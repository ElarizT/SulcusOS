from kernel.process import AgentProcess

class Parent(AgentProcess):
    name = "Parent"
    supervisor_strategy = "rest_for_one"
