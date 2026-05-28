from kernel.process import AgentProcess


class PongAgent(AgentProcess):
    name = "PongAgent"
    capabilities = ("pong", "ipc")

    async def on_message(self, message) -> None:
        if message.type == "task_request":
            self.reply(message, {"pong": True, "received": message.payload})
            self.remember({"event": "pong_reply", "correlation_id": message.correlation_id}, 1)
