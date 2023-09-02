# 📝 promptbook
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
## TODO
TODO
