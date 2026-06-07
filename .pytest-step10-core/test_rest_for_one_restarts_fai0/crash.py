from kernel.process import AgentProcess

class CrashWorker(AgentProcess):
    name = "CrashWorker"
    async def on_message(self, message):
        raise RuntimeError("boom")
