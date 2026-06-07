from kernel.process import AgentProcess

class Crash(AgentProcess):
    name = "Crash"
    async def on_start(self):
        raise RuntimeError("startup boom")
