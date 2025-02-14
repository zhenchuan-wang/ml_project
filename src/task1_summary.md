
### Describe any choices you had to make regarding the model architecture outside of the transformer backbone.
1. **Mean Pooling with Attention Mask**: Instead of using the [CLS] token or max pooling, mean pooling with an attention mask was chosen to aggregate token embeddings. This ensures that padding tokens do not affect the final sentence embedding.

2. **L2 Normalization**: The pooled embeddings are L2-normalized to ensure consistent scaling, which is useful for downstream tasks like cosine similarity.

3. **Tokenizer Configuration**: The tokenizer is configured to handle padding, truncation, and a maximum sequence length of 512 tokens, ensuring compatibility with the transformer model.