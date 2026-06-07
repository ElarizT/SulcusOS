
from kernel.process import AgentProcess

class EchoAgent(AgentProcess):
    name = "EchoAgent"
    capabilities = ("echo",)

    async def on_message(self, message):
        self.remember({"received": message.payload}, 3)
