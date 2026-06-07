from kernel.process import AgentProcess

class Parent(AgentProcess):
    name = "Parent"
    supervisor_strategy = "one_for_one"
