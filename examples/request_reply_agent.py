from kernel.process import AgentProcess


class RequestReplyAgent(AgentProcess):
    name = "RequestReplyAgent"
    capabilities = ("request-reply", "ipc")

    async def on_start(self) -> None:
        response = await self.request(100, {"question": "are you alive?"}, timeout=2.0)
        self.remember(
            {
                "event": "request_reply_complete",
                "message_type": response.type,
                "payload": response.payload,
            },
            2,
        )

    async def on_message(self, message) -> None:
        if message.type == "task_request":
            self.reply(message, {"ok": True, "echo": message.payload})
