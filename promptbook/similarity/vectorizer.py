""" Vectorize sentence(s) in type of str or list[str] with huggingface models.
Used in Similarity Plugin.
"""
import typing
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class Vectorizer:
    """ Get sentence embeddings. Object of this class are callable.
    """
    def __init__(self,
                 model_name: str = "intfloat/multilingual-e5-base",
                 batch_size: int = 64,
                 seed: int = 42):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = seed
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.batch_size = batch_size
        
        self._make_deterministic(seed=self.seed)
        
    def average_pool(self,
                     last_hidden_states: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(),
                                                     0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def __call__(self,
                 payload: typing.Union[list[str], str],
                 tqdm_on: bool = True) -> np.ndarray:
        """ Entry point to all class logic that transform
        an object to callable
        """
        if isinstance(payload, str):
            tokenizer_output = self.tokenizer(payload,
                                              max_length=512,
                                              padding=True,
                                              truncation=True,
                                              return_tensors='pt')
            output = self.model(**{k: v.to(self.device) for k, v in tokenizer_output.items()})
            embeddings = self.average_pool(output.last_hidden_state.detach().cpu(),
                                           tokenizer_output['attention_mask'].detach().cpu())
            embeddings = F.normalize(embeddings, p=2, dim=1)  # normalize embeddings

            return embeddings

        payload_embeddings = None
        
        # split payload on batches
        if tqdm_on:
            gen = tqdm(range(0, len(payload), self.batch_size))
        else:
            gen = range(0, len(payload), self.batch_size)
        for idx in gen:
            min_batch_idx, max_batch_idx = idx, min(idx+self.batch_size, len(payload))
            batch: list[str] = list(payload[min_batch_idx:max_batch_idx])
            tokenizer_output = self.tokenizer(batch,
                                              max_length=512,
                                              padding=True,
                                              truncation=True,
                                              return_tensors='pt')
            
            output = self.model(**{k:v.to(self.device) for k,v in tokenizer_output.items()})
            embeddings = self.average_pool(output.last_hidden_state.detach().cpu(),
                                           tokenizer_output['attention_mask'].detach().cpu())
            del output
            embeddings = F.normalize(embeddings, p=2, dim=1)  # normalize embeddings

            embeddings = embeddings.detach().cpu().numpy()
            
            if payload_embeddings is None:
                payload_embeddings = embeddings
            else:
                payload_embeddings = np.vstack([payload_embeddings, embeddings])
                
        return payload_embeddings

    def _make_deterministic(self, seed) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        return
