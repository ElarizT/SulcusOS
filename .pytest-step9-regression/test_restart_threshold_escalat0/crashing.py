from kernel.process import AgentProcess

class Crashing(AgentProcess):
    name = "Crashing"
    async def on_start(self):
        raise RuntimeError("startup boom")
