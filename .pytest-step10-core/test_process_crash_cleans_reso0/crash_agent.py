
from kernel.process import AgentProcess

class CrashAgent(AgentProcess):
    name = "CrashAgent"

    async def on_start(self):
        raise RuntimeError("boom")
