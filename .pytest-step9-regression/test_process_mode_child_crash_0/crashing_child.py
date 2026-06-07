
from kernel.process import AgentProcess

class CrashingChild(AgentProcess):
    name = "CrashingChild"

    async def on_start(self):
        raise RuntimeError("child boom")
