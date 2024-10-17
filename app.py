import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Load the datasets
def load_data(file_path):
    return pd.read_csv(file_path)

# Prepare the data for BERT
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Fine-tune the BERT model
def train_model(data, text_column, label_column):
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Ensure labels are encoded properly
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data[label_column].tolist())
    
    # Check number of unique labels
    num_labels = len(set(labels))
    
    # Initialize the model with the correct number of labels
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    texts = data[text_column].astype(str).tolist()
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    dataset = CustomDataset(encodings, labels)
    
    # Split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    
    # Evaluate the model
    trainer.evaluate()
    predictions, labels, _ = trainer.predict(test_dataset)
    preds = predictions.argmax(axis=1)
    accuracy = accuracy_score(labels, preds)
    report = classification_report(labels, preds)
    
    # Print accuracy and classification report
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(report)
    
    return tokenizer, model, label_encoder

# Save the model, tokenizer, and label encoder
def save_model(tokenizer, model, label_encoder):
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

# Load the model, tokenizer, and label encoder
def load_model():
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return tokenizer, model, label_encoder

# Predict function
def predict(text, tokenizer, model, label_encoder):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(predictions, dim=1).item()
    sentiment = "Positive" if label_encoder.inverse_transform([predicted_label])[0] == 1 else "Negative"
    return sentiment, predictions

# Main function for Streamlit app
def main():
    st.title("BERT Sentiment Classification")

    # Load dataset
    file_path = "C:/Users/msi00/OneDrive/Desktop/data.csv"
    data = load_data(file_path)
    
    st.write("Sample Data:")
    st.write(data.head())
    
    # Columns for user to select
    st.write("Columns in the dataset:")
    st.write(data.columns)
    
    # User input for text and label columns
    text_column = st.selectbox("Select the text column", data.columns)
    label_column = st.selectbox("Select the label column", data.columns)
    
    # Train model and tokenizer
    if st.button("Train Model"):
        with st.spinner('Training...'):
            try:
                tokenizer, model, label_encoder = train_model(data, text_column, label_column)
                # Save model and tokenizer
                save_model(tokenizer, model, label_encoder)
                st.success("Model trained and saved successfully!")
            except Exception as e:
                st.error(f"Error during training: {e}")
    
    st.header("Enter Text for Prediction")
    user_input = st.text_area("Text Input", height=150, placeholder="Type your text here...")
    
    if st.button("Predict"):
        if user_input.strip():
            with st.spinner('Predicting...'):
                try:
                    # Load the model and tokenizer
                    if os.path.exists('tokenizer.pkl') and os.path.exists('model.pkl') and os.path.exists('label_encoder.pkl'):
                        tokenizer, model, label_encoder = load_model()
                        sentiment, predictions = predict(user_input, tokenizer, model, label_encoder)
                        st.write("Predicted Sentiment:")
                        st.write(sentiment)
                        st.write("Prediction Probabilities:")
                        st.write(predictions.numpy())
                    else:
                        st.error("Model and tokenizer not found. Please train the model first.")
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
        else:
            st.warning("Please enter some text for prediction.")

if __name__ == "__main__":
    main()
