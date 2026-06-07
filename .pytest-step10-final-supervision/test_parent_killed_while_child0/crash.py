from kernel.process import AgentProcess

class Crash(AgentProcess):
    name = "Crash"
    async def on_message(self, message):
        raise RuntimeError("boom")
