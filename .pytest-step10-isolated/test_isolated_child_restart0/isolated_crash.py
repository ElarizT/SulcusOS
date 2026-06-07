from kernel.process import AgentProcess

class IsolatedCrash(AgentProcess):
    name = "IsolatedCrash"
    async def on_message(self, message):
        raise RuntimeError("isolated boom")
