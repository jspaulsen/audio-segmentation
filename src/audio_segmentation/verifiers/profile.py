import torch
from torch import Tensor


class SpeakerProfile:
    """
    Manages and represents a speaker's identity using an average embedding.

    This class provides two methods for updating the speaker's representative
    embedding (centroid):
    1.  Standard Running Average: All embeddings are weighted equally. Ideal for
        building a stable, long-term profile.
    2.  Exponential Moving Average (EMA): Gives more weight to recent embeddings,
        allowing the profile to adapt over time.

    Attributes:
        centroid (Optional[Tensor]): The representative embedding for the speaker.
        count (int): The number of embeddings included in the standard average.
        similarity_fn: The function used to compute cosine similarity.
    """
    def __init__(
        self,
        speaker_id: int,
        centroid: Tensor,
    ) -> None:
        self.speaker_id = speaker_id
        self.centroid: Tensor = centroid
        self.count: int = 1
        self.similarity_fn = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def add_embedding(self, embedding: Tensor):
        """
        Updates the centroid with a new embedding using a standard running average.
        This method is numerically stable and weights all embeddings equally.

        Args:
            embedding (Tensor): A 1D tensor of the new embedding.
        """
        if embedding.dim() != 1:
            raise ValueError("The provided embedding must be a 1D tensor.")

        self.count = self.count + 1
        self.centroid = self.centroid + (embedding - self.centroid) / self.count

    def add_embedding_ema(self, new_embedding: Tensor, alpha: float = 0.1):
        """
        Updates the centroid using an Exponential Moving Average (EMA).
        This gives more weight to recent embeddings, allowing the profile to adapt.

        Args:
            new_embedding (Tensor): A 1D tensor of the new embedding.
            alpha (float): The smoothing factor (between 0 and 1). A higher alpha
                           means faster adaptation to new embeddings.
        """
        if new_embedding.dim() != 1:
            raise ValueError("The provided embedding must be a 1D tensor.")

        self.centroid = (1 - alpha) * self.centroid + alpha * new_embedding

    def compare(
        self,
        embedding: Tensor,
        threshold: float = 0.75,
    ) -> tuple[float, bool]:
        """
        Compares an incoming embedding to the speaker's current centroid.

        Args:
            embedding (Tensor): The 1D embedding tensor to be tested.
            threshold (float): The similarity score required to be considered a match.

        Returns:
            A tuple containing:
                - The similarity score (float).
                - A boolean indicating if the score exceeds the threshold (bool).
        """
        if embedding.dim() != 1:
             raise ValueError("The incoming embedding must be a 1D tensor.")

        # Unsqueeze both tensors to shape (1, N) for the similarity function
        score_tensor: Tensor = self.similarity_fn(
            embedding.unsqueeze(0),
            self.centroid.unsqueeze(0)
        )

        score: float = score_tensor.item()
        return score, score > threshold
