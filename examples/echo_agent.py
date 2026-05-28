from kernel.process import AgentProcess


class EchoAgent(AgentProcess):
    name = "EchoAgent"
    capabilities = ("echo",)

    async def on_start(self) -> None:
        self.remember({"event": "started"}, 1)

    async def on_message(self, message) -> None:
        self.remember({"event": "message", "payload": message.payload}, 3)
