from kernel.process import AgentProcess


class IsolatedCounterAgent(AgentProcess):
    name = "IsolatedCounterAgent"
    capabilities = ("counter",)

    async def on_start(self) -> None:
        self.remember({"event": "isolated-started"}, 1)

    async def on_stop(self) -> None:
        self.remember({"event": "isolated-stopped"}, 1)
