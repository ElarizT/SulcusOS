from kernel.process import AgentProcess


class PingAgent(AgentProcess):
    name = "PingAgent"
    capabilities = ("ping", "ipc")

    async def on_start(self) -> None:
        response = await self.request(100, {"ping": True, "text": "hello from ping"}, timeout=2.0)
        self.remember(
            {
                "event": "ping_response",
                "message_type": response.type,
                "payload": response.payload,
                "correlation_id": response.correlation_id,
            },
            2,
        )
