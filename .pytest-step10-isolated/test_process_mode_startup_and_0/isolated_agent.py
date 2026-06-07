
from kernel.process import AgentProcess

class IsolatedAgent(AgentProcess):
    name = "IsolatedAgent"
    capabilities = ("isolated",)
