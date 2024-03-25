# promptbook v2.0
Use this lib to build LLM prompts with examples.

## Quick Start
### Create dataset
```python
>>> from promptbook.similarity import SimilarityPlugin
>>> # create the plugin instance
>>> sp = SimilarityPlugin(vectorizer_batch_size=50, n_trees=100)
>>> # now you need to vectorize your corpus with `sp.vectorizer`
>>> embeddings: torch.Tensor = sp.vectorizer(data["text"].values, tqdm_on=True)
>>> # basically you need payload to be a list[Any] (or any subscriptable)
>>> # you can play with format as you like
>>> payload: list = data.to_dict(orient="records")
>>> # in our example payload is list[dict], you can explicitly set what you want
>>> # to save in vector DB with `sp.dump_process_fn` ("embedding" by default)
>>> for idx in range(len(payload)):
>>>    payload[idx]["embedding"] = embeddings[idx]
>>> sp.build_storages(payload)
```
### Build prompts with existing dataset
```python
>>> sp = SimilarityPlugin()
>>> # to tune format of value returned by sp.process rewrite sp.return_format_fn
>>> # it's return `(x["text"], x["entity"], x["score"])` by default
>>> # use `permutation=True` with `noise_std=...` to get more diverse results
>>> sp.process("The king is dead, long live the king!", k=1, permutation=True, noise_std=0.01)
("Turtles have a long life", "turtles", "Positive")
```
### Process many queries in the same time
Let's assume we have a custom data we need to process batch by batch

Firstly, we need to write function to process our customo data
## Vectorization
```python
def vectorizing_strategy(vectorizer, data):
  """ Our strategy to vectorize batches"""
  embds = []
  for row in data:
    post_embd: str = vectorizer(row["post_text"])
    layer1_embd = vectorizer(row["parent_comment_text"])
    layer2_embd = vectorizer(row["comment_text"])
    embd = SIGMA * post_embd + OMEGA * layer1_embd + GAMMA * layer2_embd
    embds.append(embd)
  return embds
```
Next we can process our batch

```python
similar_items = sp.process(
    batch,
    vectorizing_strategy=vectorizing_strategy,
    strategy="centroid_outsider",
    strategy_object=None
)
```

You can pass custom `strategy_object` (inherit from `promptbook.similarity.strategies.Strategy`)
