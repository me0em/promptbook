""" 
"""
import torch


class Strategy:
    """ Parent class for all strategies
    """
    def __init__(self,
                 embeddings: torch.Tensor) -> None:
        self.embeddings = embeddings

    def run(self):
        """ Choose embeddings
        """
        raise NotImplementedError
