from kernel.process import AgentProcess

class Parent(AgentProcess):
    name = "Parent"
    async def on_message(self, message):
        raise RuntimeError("supervisor boom")
