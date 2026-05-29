from kernel.process import AgentProcess


class MemoryWriterAgent(AgentProcess):
    name = "MemoryWriterAgent"
    capabilities = ("memory-write",)
    token_budget = 12

    async def on_start(self) -> None:
        self.remember(
            {"note": "alpha project uses structured IPC"},
            token_estimate=5,
            importance=0.9,
            tags=["project", "ipc"],
        )
        self.remember(
            {"note": "beta task should be recalled later"},
            token_estimate=5,
            importance=0.6,
            tags=["task", "beta"],
        )
        self.remember(
            {"note": "low priority scratch memory"},
            token_estimate=5,
            importance=0.1,
            tags=["scratch"],
        )
