from kernel.process import AgentProcess


class MemoryRecallerAgent(AgentProcess):
    name = "MemoryRecallerAgent"
    capabilities = ("memory-recall",)

    async def on_start(self) -> None:
        project_memories = self.recall(tags=["project"], limit=5)
        beta_memories = self.recall(query="beta", limit=5)
        self.remember(
            {
                "event": "memory_recalled",
                "project_count": len(project_memories),
                "beta_count": len(beta_memories),
            },
            token_estimate=3,
            importance=0.7,
            tags=["recall-result"],
        )
