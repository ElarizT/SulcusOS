
from kernel.process import AgentProcess

class Ping(AgentProcess):
    name = "Ping"

    async def on_start(self):
        response = await self.request(100, {"ping": True}, timeout=2.0)
        if response.payload.get("pong"):
            self.send(100, {"seen": True}, message_type="event")
