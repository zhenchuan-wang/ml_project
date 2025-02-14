import torch
from transformers import AutoModel, AutoTokenizer


class SentenceTransformer(torch.nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(SentenceTransformer, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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

        return pooled_embeddings


# Test the model
model = SentenceTransformer()
sentences = ["This is a sample sentence.", "Another example sentence."]
embeddings = model(sentences)

# Check the shape of the output embeddings (should be [batch_size, embedding_size])
print(embeddings)
print(f"Embedding shape: {embeddings.shape}")
