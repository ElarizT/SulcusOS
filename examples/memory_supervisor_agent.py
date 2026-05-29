from kernel.process import AgentProcess


class MemorySupervisorAgent(AgentProcess):
    name = "MemorySupervisorAgent"
    capabilities = ("memory-supervision",)
    supervisor_strategy = "one_for_one"

    async def on_start(self) -> None:
        await self.spawn_child("examples/memory_writer_agent.py", restart_policy="permanent")
        self.remember({"event": "memory_supervisor_started"}, tags=["supervisor"])

    async def on_message(self, message) -> None:
        if message.type == "event":
            self.remember({"supervision_event": message.payload}, tags=["supervision"])
