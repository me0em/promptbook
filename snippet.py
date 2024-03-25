""" Try to run everything
to make sure all works as should be
"""
import json
import pandas as pd

import torch

from promptbook.similarity import SimilarityPlugin


def load_pl1l2(df_path="../datasets/subsamples_with_prompts.json"):
    with open(df_path, "r", encoding="utf-8") as file:
        df = json.load(file)

    df = pd.DataFrame(df)

    pl1l2 = df[df.parent_comment_text.notna()]

    del df
    return pl1l2

def build_embeddings(sp: SimilarityPlugin,
                     pl1l2: pd.DataFrame) -> tuple[torch.Tensor]:

    print("vectorize posts...")
    post_embeddings: torch.Tensor = sp.vectorizer(
        pl1l2["post_text"].values,
        tqdm_on=True
    )

    print("vectorize layer-1 comments...")
    layer1_embeddings: torch.Tensor = sp.vectorizer(
        pl1l2["parent_comment_text"].values,
        tqdm_on=True
    )

    print("vectorize layer-2 comments...")
    layer2_embeddings: torch.Tensor = sp.vectorizer(
        pl1l2["comment_text"].values,
        tqdm_on=True
    )

    return post_embeddings, layer1_embeddings, layer2_embeddings


if __name__ == "__main__":
    sp = SimilarityPlugin(vectorizer_batch_size=50,
                          n_trees=100,
                          storage_path="/Users/alexander/Desktop/meow",
                          instance_name="test")

    # Build embeddings
    # __________________________________
    pl1l2 = load_pl1l2()
    post_embeddings, layer1_embeddings, layer2_embeddings = build_embeddings(sp, pl1l2)

    # Result embedding := \sigma * P + \omega * L1 + \gamma * L2
    SIGMA = 0.1
    OMEGA = 0.3
    GAMMA = 0.6
    # embeddings should have shape (n, 768), where is number of objects
    embeddings = SIGMA * post_embeddings + \
        OMEGA * layer1_embeddings + \
        GAMMA * layer2_embeddings

    print(f"{embeddings.shape=}")

    # Build storage
    # __________________________________
    payload: list = pl1l2.to_dict(orient="records")
    for idx, _ in enumerate(payload):
        payload[idx]["embedding"] = embeddings[idx]

    print("Building the storage...")
    sp.build_storages(payload)
    print("The storage has been build")

    sp.return_format_fn = lambda x: (x["post_text"],
                                     x["comment_text"],
                                     x["parent_comment_text"])

    similar_item = sp.process(
        "Вы полная хуйня, никогжда вами не пользуюсь, худший банк",
        k=1,
        permutation=True,
        noise_std=0.01
    )

    print(similar_item)
