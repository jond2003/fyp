import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
from global_vars import BLOGS_TRAIN_PATH, BERT_PATH, TRAINING_IMG_PATH

# ensure CUDA operations raise errors upon failure
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# custom dataset class for text classification using BERT
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
        # tokenize the text at the given index
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}


# BERT-based classification model
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)  # dropout for regularisation
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)  # final classification layer

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # get [CLS] token representation
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits


# training loop
def train(model, data_loader, optimizer, scheduler, device):
    model.train()  # set model to training mode
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)  # compute loss
        loss.backward()
        optimizer.step()
        scheduler.step()  # update learning rate

# evaluation loop
def evaluate(model, data_loader, device):
    model.eval()  # set model to evaluation mode
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)


# load training data
def load_blog_data(data_file):
    df = pd.read_csv(data_file)
    blogs = df['text'].tolist()
    labels = df['label'].tolist()
    return blogs, labels


def train_classifier():
    blogs, labels = load_blog_data(BLOGS_TRAIN_PATH)

    # model parameters
    bert_model_name = 'bert-base-uncased'
    num_classes = len(set(labels))
    max_length = 128
    batch_size = 16
    num_epochs = 3
    learning_rate = 2e-5

    val_accuracies = []  # accuracies per epoch

    # split data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(blogs, labels, test_size=0.2, random_state=42)

    # initialise tokenizer and datasets
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # initialise model, optimizer, and learning rate scheduler
    model = BERTClassifier(bert_model_name, num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # training and evaluation loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_dataloader, optimizer, scheduler, device)
        accuracy, report = evaluate(model, val_dataloader, device)
        val_accuracies.append({'Epoch': epoch + 1, 'Validation Accuracy': accuracy})
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(report)

    # save trained model
    torch.save(model.state_dict(), BERT_PATH)

    # display validation accuracy table
    accuracy_df = pd.DataFrame(val_accuracies)
    print("\nValidation Accuracy per Epoch:")
    print(accuracy_df.to_string(index=False))

    # plot and save validation accuracy as an image
    epochs = [entry['Epoch'] for entry in val_accuracies]
    accuracies = [entry['Validation Accuracy'] for entry in val_accuracies]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, marker='o', linestyle='-', color='blue')
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.grid(True)
    plt.savefig(TRAINING_IMG_PATH)  # save image
    plt.show()