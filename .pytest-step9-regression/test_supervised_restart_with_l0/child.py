from kernel.process import AgentProcess

class Child(AgentProcess):
    name = "Child"
    memory_restore_policy = "latest_snapshot"
    async def on_message(self, message):
        if message.payload.get("cmd") == "crash":
            raise RuntimeError("boom")
