import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# Load dataset
df = pd.read_csv("dataset.csv")

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])  # human=0, ai=1

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2
)

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=256)

train_enc = tokenize(train_texts)
test_enc = tokenize(test_texts)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_enc, train_labels)
test_dataset = Dataset(test_enc, test_labels)

# Model
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=2
)

# Training
args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Save model
model.save_pretrained("model")
tokenizer.save_pretrained("model")
python train.py
