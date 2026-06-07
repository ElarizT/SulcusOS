
from kernel.process import AgentProcess

class Requester(AgentProcess):
    name = "Requester"

    async def on_start(self):
        response = await self.request(100, {"ping": "hello"}, timeout=0.05)
        self.remember({"code": response.payload["code"]}, 1)
