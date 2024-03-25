""" Choose centroid and the most remote point from centroid
"""
import torch
from .strategy import Strategy


class CentroidAndOutsiderStrategy(Strategy):
    """ Choose centroid and the most remote point from centroid
    """
    def __init__(self, embeddings: torch.Tensor):
        super().__init__(embeddings)

    def run(self):
        return self.calculate_anchor_points()

    def calculate_anchor_points(self) -> dict[str, int]:
        """ metric is euclidian
        """
        centroid_idx: int = 0
        outsider_idx: int = 0
        current_centroid_delta: int = 1e8
        current_outsider_delta: int = 0
        center_of_mass: torch.Tensor = self.embeddings.mean(dim=0)

        # euclidian
        metric: callable = lambda a, b: torch.sqrt(torch.sum((a-b)**2))

        for idx, emb in enumerate(self.embeddings):
            if (delta := metric(emb, center_of_mass)) < current_centroid_delta:
                current_centroid_delta = delta
                centroid_idx = idx

            if (delta := metric(emb, center_of_mass)) > current_outsider_delta:
                current_outsider_delta = delta
                outsider_idx = idx

        return {"centroid_idx": centroid_idx, "outsider_idx": outsider_idx}
