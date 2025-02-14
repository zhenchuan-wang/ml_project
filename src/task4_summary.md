### Assumptions and Decisions in the Multi-Task Learning (MTL) Framework:

1. **Shared Backbone**:
   - Assumption: The transformer backbone (e.g., BERT) provides general-purpose sentence embeddings useful for both tasks.
   - Decision: The backbone is shared across tasks to leverage transfer learning and reduce computational cost.

2. **Task-Specific Heads**:
   - Assumption: Each task requires specialized layers to adapt the shared embeddings to its specific output space.
   - Decision: Separate heads are added for sentence classification (Task A) and sentiment analysis (Task B).

3. **Loss Calculation**:
   - Assumption: Both tasks contribute equally to the model's learning.
   - Decision: The total loss is a simple sum of individual task losses (`loss_task_a + loss_task_b`). Weighted loss could be used if tasks have different importance.

4. **Freezing Components**:
   - Assumption: Freezing parts of the model can prevent overfitting or retain pre-trained knowledge.
   - Decision: Options to freeze the backbone, Task A head, or Task B head are provided for flexibility.

5. **Metrics**:
   - Assumption: Weighted averages for precision, recall, and F1-score are sufficient for evaluating multi-class tasks.
   - Decision: Metrics are calculated using `average="weighted"` for simplicity. Task-specific metrics can be added if needed.

6. **Data Handling**:
   - Assumption: Input data for both tasks is available in the same dataset.
   - Decision: A unified `Dataset` class is used to handle sentences and labels for both tasks.
