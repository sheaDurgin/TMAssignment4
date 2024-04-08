import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import numpy as np

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
    print("read data")
    all_labels = y_train + y_val + y_test
    label_encoder.fit(all_labels)
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)

    train_dataset = tokenize_data(X_train, y_train_encoded)
    val_dataset = tokenize_data(X_val, y_val_encoded)
    test_dataset = tokenize_data(X_test, y_test_encoded)

    model_path = '/home/shea.durgin/netstore1/distilroberta_results'

    # Early Stopping
    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
    hyperparameters = [16, 32, 64, 128] # set this to your values
    for param in hyperparameters:
		model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=len(set(y_train)))
        print(f"using {param}")
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=model_path,
            learning_rate=2e-5,
            weight_decay=0.01,
            load_best_model_at_end = True,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=4,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy='epoch',
            save_strategy='epoch'
        )

        # Define Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[early_stopping],
        )

        # Train the model
        trainer.train()
        # trainer.save_model(model_path)

        # Evaluate the model
        test_results = trainer.predict(test_dataset)

        # Extract predicted labels
        predicted_labels = np.argmax(test_results.predictions, axis=1)

        # Decode predicted labels
        predicted_labels = label_encoder.inverse_transform(predicted_labels)

        # Calculate F1 score for each class and total F1 score
        f1_scores_per_class = f1_score(y_test, predicted_labels, average=None)
        total_f1_score = f1_score(y_test, predicted_labels, average='weighted')

        # Print F1 scores
        for label, f1_score_class in zip(label_encoder.classes_, f1_scores_per_class):
            print(f"F1 score for class {label}: {f1_score_class}")

        print(f"Total F1 score: {total_f1_score}")
