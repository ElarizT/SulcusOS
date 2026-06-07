
from kernel.process import AgentProcess

class Ping(AgentProcess):
    name = "Ping"

    async def on_start(self):
        response = await self.request(100, {"ping": "hello"}, timeout=1.0)
        self.remember({"response": response.payload, "correlation_id": response.correlation_id}, 1)
