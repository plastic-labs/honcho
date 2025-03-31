# Theory of Mind Inference
[Theory of Mind](https://blog.plasticlabs.ai/blog/Theory-of-Mind-Is-All-You-Need) is a core principle behind Honcho: we believe that enabling AI agents to reason about users' mental states is essential if we want them to successfully act on our behalf.

Honcho currently features three different modules for theory of mind inference:
- `conversational.py`: Inspired by our work on [metanarrative prompting](https://blog.plasticlabs.ai/blog/Agent-Identity). Uses a metanarrative prompt for both ToM inference and generating a user representation.
- `single_prompt.py`: A more conventional and straightforward approach that specifies in a single system prompt what it wants the LLM to output.
- `long_term.py`: Formats a theory of mind inference and a series of long-term facts into a user representation.

The current setup works as follows:
- We extract facts from incoming messages using the code in `src.deriver.consumer`.
- These messages get added to the protected `honcho` user collection using the `CollectionEmbeddingStore` in `src.deriver.tom.embeddings`.
- The dialectic endpoint, in `src.agent`, retrieves long-term facts from this store that are relevant to the query, and runs the ToM inference in `src.deriver.tom.single_prompt` to generate a prediction of the user's short-term mental state.
- The retrieved long-term facts and the short-term ToM inference are combined into a user representation. By default, this is done using a simple f-string, but they can optionally be combined using a separate inference, which would use `src.deriver.tom.long_term`.