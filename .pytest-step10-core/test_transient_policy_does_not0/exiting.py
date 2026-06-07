from kernel.process import AgentProcess

class Exiting(AgentProcess):
    name = "Exiting"
    async def run(self):
        return
