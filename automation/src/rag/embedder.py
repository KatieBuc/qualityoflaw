from openai import AzureOpenAI, OpenAI


class AzureEmbedder:
    def __init__(self, deployment: str, client: OpenAI | AzureOpenAI, batch_size: int = 16):
        self.deployment = deployment
        self.client = client
        self.batch_size = batch_size

    @classmethod
    def from_env(cls, *, batch_size: int = 16) -> "AzureEmbedder":
        from automation.src.llm.client import get_azure_client, get_embedding_deployment

        return cls(
            deployment=get_embedding_deployment(),
            client=get_azure_client(),
            batch_size=batch_size,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(model=self.deployment, input=batch)
            embeddings.extend(item.embedding for item in response.data)
        return embeddings
