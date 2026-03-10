import os
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

print("Loading GoEmotions parquet dataset locally...")

dataset = load_dataset(
    "parquet",
    data_files="data/text/goemotions/train-00000-of-00001.parquet"
)
                                        
df = dataset["train"].to_pandas()

print("Columns in dataset:", df.columns)
print("Total samples:", len(df))

# Labels column contains list of emotion indices
# Example: [3, 15]

# Convert multi-label to single label by taking first label
df = df[df["labels"].map(len) > 0]  # remove empty labels
df["label"] = df["labels"].apply(lambda x: x[0])

df = df[["text", "label"]]

# Get number of unique labels 
num_labels = df["label"].nunique()

print("Number of emotion classes:", num_labels)

# Split dataset
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels
)

training_args = TrainingArguments(
    output_dir="models/text_emotion",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_steps=100,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

print("Starting training...")
trainer.train()

print("Saving model...")
trainer.save_model("models/text_emotion")
tokenizer.save_pretrained("models/text_emotion")

print("Training complete.")

