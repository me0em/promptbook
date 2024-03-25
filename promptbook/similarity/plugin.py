""" This module can be connected to the Prompt Engine
as a plugin.

Similarity Plugin take test sentence and find the nearest
(with respect to different metrics) examples from train
dataset to inject them in the prompt afterwards.

Annoy library using here cuz it's very fast on benchmarks
and used in top-tier highload products.

Example of creating database:
>>> sp = SimilarityPlugin(vectorizer_batch_size=50, n_trees=100)
>>> # data is pd.DataFrame like {"text": [...], "score": [...], "entity": [...]},
>>> # by default, but you can tune format with sp.return_format_fn and sp.dump_process_fn
>>> # also you can turn on vectorizer progress bar with tqdm_on=True
>>> embeddings: torch.Tensor = sp.vectorizer(data["text"].values, tqdm_on=True)
>>> # basically we need payload to be a list[Any] (or another words, to be subscriptable)
>>> # you can play with format as you like
>>> payload: dict = data.to_dict(orient="records")
>>> for idx in range(len(payload)):
>>>    payload[idx]["embedding"] = embeddings[idx]
>>> sp.build_storages(payload)

Example of working with created database:
>>> sp = SimilarityPlugin()
>>> # to tune format of value returned by sp.process rewrite sp.return_format_fn
>>> sp.process("The king is dead, long live the king!", k=1)
("Turtles have a long life", "turtles", "Positive")
>>>
"""
from typing import Optional, Iterable
import os
import pickle

import torch
import einops
from annoy import AnnoyIndex

from .vectorizer import Vectorizer
from .strategies import Strategy, CentroidAndOutsiderStrategy


class SimilarityPlugin:
    """ The purpose of this class is to take test sentence and
    find the nearest (w.r.t. semantic similarity) examples from
    train embeddings dataset to inject them in the prompt
    """
    def __init__(self,
                 # now storage_path is mandatory
                 storage_path: str,
                 # now you can pass vectorizer from outside
                 vectorizer: Optional[Vectorizer] = None,
                 instance_name: Optional[str] = "unnamed",
                 vectorizer_batch_size: int = 32,
                 n_trees: int = 100) -> None:
        if vectorizer:
            self.vectorizer = vectorizer
        else:
            self.vectorizer = Vectorizer(batch_size=vectorizer_batch_size)
        self.embeddings_size: int = 768
        self.metric = "angular"
        self.n_trees = n_trees
        self.storage_path = storage_path

        self.instance_name = instance_name
        # storage with real items
        self.saved_corpus_name = f"saved_embeddings_{self.instance_name}.pickle"
        # storage with embeddings tree
        self.saved_index_name = f"embedding_index_{self.instance_name}.ann"
        # real items, formatted as (text, payload as dict)
        self.corpus: list[dict] = None

        self.index = None

        self.return_format_fn = lambda x: (x["text"], x["entity"], x["score"])
        self.dump_process_fn = lambda x: x["embedding"]

    def _load_corpus(self) -> list[dict]:
        """ Load source corpus with permanent item ids
        to be able to get raw text by id
        """
        path = os.path.join(self.storage_path, self.saved_corpus_name)
        with open(path, "rb") as file:
            self.corpus = pickle.load(file)

    def _dump_corpus(self) -> None:
        """ Dump source to pickle file. Use after
        adding new items
        """
        path = os.path.join(self.storage_path, self.saved_corpus_name)
        with open(path, "wb") as file:
            pickle.dump(self.corpus, file)

    def build_storages(self, data: list) -> None:
        """ Create annoy index from given embeddings and dump on disk
        """
        # Save embeddings to Annoy database
        embeddings = [self.dump_process_fn(i) for i in data]
        self.index = AnnoyIndex(self.embeddings_size, metric=self.metric)
        for i, embd in enumerate(embeddings):
            self.index.add_item(i, embd)
        self.index.build(n_trees=self.n_trees)

        # save embeddings
        path_to_save = os.path.join(self.storage_path, self.saved_index_name)
        if os.path.isdir(self.storage_path) is False:
            print(f"Try to create directory at {self.storage_path}")
            os.mkdir(self.storage_path)
        self.index.save(path_to_save)

        # Save other data to pickle file
        for idx, _ in enumerate(data):
            del data[idx]["embedding"]
        self.corpus = data
        self._dump_corpus()

    def load_index(self) -> AnnoyIndex:
        """ Load annoy index from the disk
        """
        index_path = os.path.join(self.storage_path, self.saved_index_name)
        index = AnnoyIndex(self.embeddings_size, self.metric)
        index.load(index_path)

        return index

    def add_noise(self, tensor, noise_std):
        """ Add noise to the vector to get more divorse results
        """
        noise = torch.randn(tensor.shape) * noise_std
        noisy_tensor = tensor + noise

        return noisy_tensor

    def process(self,
                data: str | Iterable,
                **kwargs):
        """Routes data to the appropriate processing method based on its type.

        This method dynamically directs the input to either `process_one` or `process_many`
        depending on whether the input is a single string or an iterable of strings, respectively.

        Args:
            data (str | Iterable): The data to be processed.
                Can be a single string or an iterable of strings.
            **kwargs: Arbitrary keyword arguments that are forwarded
                to the processing method.

        Returns:
            The result of the processing, which varies based on the
                input type and the specific
            processing methods implemented by `process_one` and `process_many`.

        Raises:
            TypeError: If `data` is neither a string nor an iterable of strings.

        """
        if isinstance(data, str):
            return self.process_one(data, **kwargs)
        if hasattr(data, '__iter__'):
            return self.process_many(data, **kwargs)

        raise TypeError("data should be str or Iterable[str]")

    def process_one(self,
                    data: str,
                    k: int = 3,
                    permutation: bool = True,
                    noise_std: float = 0.01) -> list[dict]:
        """ Find k nearest neighbors by given target embedding. Items
        stores as list of dict by default, but you can extract what you
        want with in a format you want with self.return_format_fn function.
        """
        if self.index is None:
            self.index = self.load_index()

        emb = self.vectorizer(data)
        emb = einops.rearrange(emb, "h w -> (h w)")

        if permutation:
            emb = self.add_noise(emb, noise_std)

        neighbor_ids = self.index.get_nns_by_vector(emb, k, include_distances=False)

        if self.corpus is None:
            self._load_corpus()

        neighbors = [self.corpus[idx] for idx in neighbor_ids]
        formatted_neighbors = [self.return_format_fn(i) for i in neighbors]

        return formatted_neighbors

    def process_many(self,
                     data: Iterable[str],
                     strategy: Optional[str],
                     strategy_object: Optional[Strategy],
                     vectorizing_strategy: callable) -> list[dict]:
        """ Find k nearest neighbors by given target embedding. Items
        stores as list of dict by default, but you can extract what you
        want with in a format you want with self.return_format_fn function.
        """
        if self.index is None:
            self.index = self.load_index()

        embds: list[torch.Tensor] = self.vectorize_many(
            vectorizing_strategy,
            data
        )

        embds: torch.Tensor = torch.cat(embds)

        # You can use different strategies to pick a points
        # which will be processed in 'find a similar' task
        #
        # Currently you can use ready-made strategies by
        # passing string arg `strategy`:
        # - centroid_outsider: pick centroid and the most remote point
        #
        # Or you can pass your own strategy with arg `strategy_object`,
        # for that you should inherit from promptbook.similarity.strategies.Strategy
        match strategy:
            case "centroid_outsider":  # default
                used_strategy: Strategy = CentroidAndOutsiderStrategy(embds)
                indexes = used_strategy.run().values()

            case None:
                if strategy_object is None:
                    raise ValueError("Use strategy or strategy_object arguments")
                raise NotImplementedError("TODO: custom strategies")

        neighbor_ids = []
        for idx in indexes:
            emb = embds[idx]
            neighbor_id = self.index.get_nns_by_vector(emb, n=1, include_distances=False)
            neighbor_ids.extend(neighbor_id)

        if self.corpus is None:
            self._load_corpus()

        neighbors = [self.corpus[idx] for idx in neighbor_ids]
        formatted_neighbors = [self.return_format_fn(i) for i in neighbors]

        return formatted_neighbors

    def vectorize_many(self, vectorizing_strategy: callable, data: Iterable) -> torch.Tensor:
        """ Wrap user vectorizing_strategy function to pass self.vectorizer inside
        """
        return vectorizing_strategy(self.vectorizer, data)
