import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class MultiTaskSentenceTransformer(nn.Module):
    def __init__(
        self, model_name="bert-base-uncased", num_classes_task_a=3, num_classes_task_b=2
    ):
        super(MultiTaskSentenceTransformer, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Task A: Sentence Classification Head
        self.task_a_head = nn.Linear(self.model.config.hidden_size, num_classes_task_a)

        # Task B: Sentiment Analysis Head
        self.task_b_head = nn.Linear(self.model.config.hidden_size, num_classes_task_b)

    def forward(self, input_sentences):
        # Tokenize input sentences
        inputs = self.tokenizer(
            input_sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Get the output from the transformer model
        outputs = self.model(**inputs)

        # Mean pooling with attention mask
        token_embeddings = outputs.last_hidden_state
        attention_mask = (
            inputs["attention_mask"]
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        pooled_embeddings = torch.sum(
            token_embeddings * attention_mask, 1
        ) / torch.clamp(attention_mask.sum(1), min=1e-9)

        # L2 normalization
        pooled_embeddings = torch.nn.functional.normalize(pooled_embeddings, p=2, dim=1)

        # Task A: Sentence Classification
        logits_task_a = self.task_a_head(pooled_embeddings)

        # Task B: Sentiment Analysis
        logits_task_b = self.task_b_head(pooled_embeddings)

        return logits_task_a, logits_task_b


# Test the model
if __name__ == "__main__":
    # Test the model
    model = MultiTaskSentenceTransformer(num_classes_task_a=3, num_classes_task_b=2)
    sentences = ["This is a sample sentence.", "Another example sentence."]
    logits_task_a, logits_task_b = model(sentences)

    print("Task A Logits (Sentence Classification):", logits_task_a)
    print("Task B Logits (Sentiment Analysis):", logits_task_b)
