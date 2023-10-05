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
import os
import pickle
import torch
import einops
from annoy import AnnoyIndex

from .vectorizer import Vectorizer


class MetadataStorage:
    """
    Represents raw text with additional metadata. Serialized and deserialized
    to/from disk using pickle. Each item in MetadataStorage is associated with
    a corresponding embedding in the vector database.
    """

    def __init__(self, storage_path: str, title: str):
        """
        Initialize a MetadataStorage instance.

        Args:
            storage_dir (str): The path to the storage dir with all DataStorage items.
            title (str): The title of the specific DataStorage item.
        """
        self.storage_dir = storage_path
        self.title = title
        self.data = {}

    def dump(self):
        """
        Serialize and dump the metadata to the storage file.
        """
        with open(self.storage_path, 'wb') as file:
            pickle.dump(self.data, file)

    def load(self):
        """
        Load and deserialize the metadata from the storage file.
        """
        try:
            with open(self.storage_path, 'rb') as file:
                self.data = pickle.load(file)
        except FileNotFoundError:
            # Handle the case where the file doesn't exist
            self.data = {}

    def add_item(self, item_name: str, embedding: list):
        """
        Add an item to the metadata storage with the associated embedding.

        Args:
            item_name (str): The name of the item.
            embedding (list): The embedding vector associated with the item.
        """
        self.data[item_name] = embedding

    def get_embedding(self, item_name: str):
        """
        Retrieve the embedding vector associated with an item.

        Args:
            item_name (str): The name of the item.

        Returns:
            list: The embedding vector associated with the item.
        """
        return self.data.get(item_name)


class EmbeddingsStorage:
    """ Each item from EmbeddingsStorage associates with the corresponding metadata from MetadataStorage.
    Serialized and deserialized in disk with Annoy database.
    """
    def __init__(self):
        pass

    def dump(self):
        pass

    def load(self):
        
        
class DataStorage:
    ...


class SimilarityPlugin:
    """ The purpose of this class is to take test sentence and
    find the nearest (w.r.t. semantic similarity) examples from
    train embeddings dataset to inject them in the prompt
    """
    def __init__(self,
                 vectorizer_batch_size: int = 32,
                 n_trees: int = 100) -> None:
        self.vectorizer = Vectorizer(batch_size=vectorizer_batch_size)
        self.embeddings_size: int = 768
        self.metric = "angular"
        self.n_trees = n_trees

        self.storage_path = os.path.join(os.path.dirname(__file__), "storage")
        self.saved_corpus_name = "saved_embeddings.pickle"  # storage with real items
        self.saved_index_name = "embedding_index.ann"  # storage with embeddings tree
        self.corpus: list[dict] = None  # real items, formatted as (text, payload as dict)

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
        path_to_save = os.path.join(self.storage_path, self.saved_index_name)
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
