### Describe the changes made to the architecture to support multi-task learning.

1. **Shared Backbone**: The transformer backbone (e.g., BERT) is shared across tasks to extract general-purpose sentence embeddings.
2. **Task-Specific Heads**:
   - **Task A (Sentence Classification)**: A fully connected layer with softmax activation is added to classify sentences into predefined classes.
   - **Task B (Sentiment Analysis)**: Another fully connected layer with softmax activation is added to predict sentiment labels (e.g., positive, negative, neutral).
3. **Multi-Task Loss**: A weighted sum of the individual task losses is used to train the model, allowing it to optimize for both tasks simultaneously.

