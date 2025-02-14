import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from task2 import MultiTaskSentenceTransformer


# Dummy Dataset for testing purposes
class MultiTaskDataset(Dataset):
    def __init__(self, sentences, labels_task_a, labels_task_b):
        self.sentences = sentences
        self.labels_task_a = labels_task_a
        self.labels_task_b = labels_task_b

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            "sentence": self.sentences[idx],
            "label_task_a": self.labels_task_a[idx],
            "label_task_b": self.labels_task_b[idx],
        }


def freeze_components(
    model, freeze_backbone=False, freeze_task_a=False, freeze_task_b=False
):
    """Freeze specific components of the model."""
    if freeze_backbone:
        for param in model.model.parameters():
            param.requires_grad = False
    if freeze_task_a:
        for param in model.task_a_head.parameters():
            param.requires_grad = False
    if freeze_task_b:
        for param in model.task_b_head.parameters():
            param.requires_grad = False


def calculate_metrics(preds, labels, task_name, num_classes):
    """Calculate precision, recall, and F1-score."""
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    print(f"{task_name} Metrics:")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    return precision, recall, f1


# Training loop
def train_model(
    model,
    dataloader,
    optimizer,
    num_epochs=3,
    freeze_backbone=False,
    freeze_task_a=False,
    freeze_task_b=False,
):
    model.train()  # Set model to training mode
    freeze_components(model, freeze_backbone, freeze_task_a, freeze_task_b)

    for epoch in range(num_epochs):
        total_loss = 0
        total_accuracy_task_a = 0
        total_accuracy_task_b = 0
        all_preds_task_a = []
        all_labels_task_a = []
        all_preds_task_b = []
        all_labels_task_b = []

        for batch in dataloader:
            sentences = batch["sentence"]
            labels_task_a = batch["label_task_a"]
            labels_task_b = batch["label_task_b"]

            # Forward pass
            optimizer.zero_grad()  # Zero the gradients
            classification_output, sentiment_output = model(sentences)

            # Calculate loss for Task A (classification)
            loss_task_a = F.cross_entropy(
                classification_output, torch.tensor(labels_task_a, dtype=torch.long)
            )

            # Calculate loss for Task B (sentiment analysis)
            loss_task_b = F.cross_entropy(
                sentiment_output, torch.tensor(labels_task_b, dtype=torch.long)
            )

            # Total loss is a sum of individual task losses (could also use weighted sum)
            total_loss_batch = loss_task_a + loss_task_b

            # Backward pass
            total_loss_batch.backward()
            optimizer.step()

            # Metrics: Accuracy, Precision, Recall, F1-Score
            _, preds_task_a = torch.max(classification_output, dim=1)
            _, preds_task_b = torch.max(sentiment_output, dim=1)

            accuracy_task_a = (
                (preds_task_a == torch.tensor(labels_task_a)).float().mean()
            )
            accuracy_task_b = (
                (preds_task_b == torch.tensor(labels_task_b)).float().mean()
            )

            total_accuracy_task_a += accuracy_task_a.item()
            total_accuracy_task_b += accuracy_task_b.item()
            total_loss += total_loss_batch.item()

            # Collect predictions and labels for additional metrics
            all_preds_task_a.extend(preds_task_a.tolist())
            all_labels_task_a.extend(labels_task_a)
            all_preds_task_b.extend(preds_task_b.tolist())
            all_labels_task_b.extend(labels_task_b)

        # Calculate additional metrics
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Loss: {total_loss / len(dataloader):.4f}")
        print(f"  Accuracy Task A: {total_accuracy_task_a / len(dataloader):.4f}")
        print(f"  Accuracy Task B: {total_accuracy_task_b / len(dataloader):.4f}")

        # Task A Metrics (Binary Classification)
        precision_task_a, recall_task_a, f1_task_a = calculate_metrics(
            all_preds_task_a, all_labels_task_a, "Task A", num_classes=2
        )

        # Task B Metrics (Multi-class Sentiment Analysis)
        precision_task_b, recall_task_b, f1_task_b = calculate_metrics(
            all_preds_task_b, all_labels_task_b, "Task B", num_classes=3
        )

        print("-" * 50)


if __name__ == "__main__":
    # Example hypothetical data (task A: sentence classification, task B: sentiment analysis)
    sentences = [
        "This is a good example.",
        "This is a bad example.",
        "I love it!",
        "I hate it.",
    ]
    labels_task_a = [1, 0, 1, 0]  # Binary classification: 1 (positive), 0 (negative)
    labels_task_b = [2, 0, 1, 0]  # Sentiment: 0 (negative), 1 (neutral), 2 (positive)

    # Initialize Dataset and DataLoader
    dataset = MultiTaskDataset(sentences, labels_task_a, labels_task_b)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Initialize model with correct number of classes for Task B
    model = MultiTaskSentenceTransformer(
        num_classes_task_a=2, num_classes_task_b=3
    )  # Task B has 3 classes
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Start the training loop with freezing options
    train_model(
        model,
        dataloader,
        optimizer,
        num_epochs=3,
        freeze_backbone=True,  # Freeze the transformer backbone
        freeze_task_a=False,  # Train Task A head
        freeze_task_b=False,  # Train Task B head
    )
