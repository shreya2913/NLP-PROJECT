import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder

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
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    
    return tokenizer, model, label_encoder

# Predict function
def predict(text, tokenizer, model, label_encoder):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(predictions, dim=1).item()
    return label_encoder.inverse_transform([predicted_label])[0], predictions

# Main function for Streamlit app
def main():
    st.title("BERT Text Classification")

    # Upload dataset
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            data = load_data(uploaded_file)
            st.write("Sample Data:")
            st.write(data.head())
            
            # Display columns for user to select
            st.write("Columns in the uploaded file:")
            st.write(data.columns)
            
            # User input for text and label columns
            text_column = st.selectbox("Select the text column", data.columns)
            label_column = st.selectbox("Select the label column", data.columns)
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            return
        
        # Train model and tokenizer
        if st.button("Train Model"):
            with st.spinner('Training...'):
                try:
                    tokenizer, model, label_encoder = train_model(data, text_column, label_column)
                    # Store model and tokenizer in session state
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model = model
                    st.session_state.label_encoder = label_encoder
                    st.success("Model trained successfully!")
                except Exception as e:
                    st.error(f"Error during training: {e}")
        
        st.header("Enter Text for Prediction")
        user_input = st.text_area("Text Input", height=150, placeholder="Type your text here...")
        
        if st.button("Predict"):
            if user_input.strip():
                with st.spinner('Predicting...'):
                    try:
                        # Access the model and tokenizer from session state
                        tokenizer = st.session_state.tokenizer
                        model = st.session_state.model
                        label_encoder = st.session_state.label_encoder
                        
                        predicted_label, predictions = predict(user_input, tokenizer, model, label_encoder)
                        st.write("Predicted Label:")
                        st.write(predicted_label)
                        st.write("Prediction Probabilities:")
                        st.write(predictions.numpy())
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
            else:
                st.warning("Please enter some text for prediction.")
    else:
        st.info("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
