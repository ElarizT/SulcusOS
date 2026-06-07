from kernel.process import AgentProcess

class Temporary(AgentProcess):
    name = "Temporary"
    async def on_start(self):
        raise RuntimeError("no restart")
