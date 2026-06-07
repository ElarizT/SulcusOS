
from kernel.process import AgentProcess

class Pong(AgentProcess):
    name = "Pong"

    async def on_message(self, message):
        self.reply(message, {"pong": True})
