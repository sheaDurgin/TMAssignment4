import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def tokenize_data(X, y, max_length=512):
    encodings = tokenizer(X, padding=True, truncation=True, max_length=max_length)
    return CustomDataset(encodings, y)

tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
label_encoder = LabelEncoder()

def get_tsv_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df['lyric'] = df['lyric'].str.lower().str.replace('[^\w\s]', '')

    X = df['lyric'].tolist()
    y = df['genre'].tolist()

    return X, y

if __name__ == '__main__':
    train_file_path = 'train.tsv'
    val_file_path = 'validation.tsv'
    test_file_path = 'test.tsv'
    X_train, y_train = get_tsv_data(train_file_path)
    X_val, y_val = get_tsv_data(val_file_path)
    X_test, y_test = get_tsv_data(test_file_path)

    all_labels = y_train + y_val + y_test
    label_encoder.fit(all_labels)
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)

    train_dataset = tokenize_data(X_train, y_train_encoded)
    val_dataset = tokenize_data(X_val, y_val_encoded)
    test_dataset = tokenize_data(X_test, y_test_encoded)

    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=len(set(y_train)))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        logging_dir='./logs',
        logging_steps=100,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print('training')

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    print(eval_results)
    
