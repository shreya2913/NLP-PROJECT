import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load the dataset directly from the 'data.csv' file
df = pd.read_csv("C:/Users/msi00/OneDrive/Desktop/data.csv")  # Ensure 'data.csv' is in the same directory

# Ensure labels are in range [0, 1]
df['sentiment_encoded'] = df['sentiment_encoded'].apply(lambda x: 1 if x == 1 else 0)

# Prediction function
def predict_sentiment(review):
    model.eval()
    inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return "POSITIVE" if prediction == 1 else "NEGATIVE"

# Streamlit app
def main():
    st.title("BERT Text Classification")

    # Display data with predictions
    st.subheader("Dataset with Sentiment Predictions")
    
    # Perform predictions for all reviews in the dataset
    df['Predicted Sentiment'] = df['reviews'].apply(predict_sentiment)
    
    # Display the updated DataFrame
    st.write(df)

    # Option to enter custom review for prediction
    st.subheader("Enter a Custom Review for Prediction")
    review = st.text_area("Enter a shoe review:")

    if st.button("Predict Sentiment"):
        if review.strip():
            sentiment = predict_sentiment(review)
            st.write(f"**Predicted Sentiment:** {sentiment}")
        else:
            st.warning("Please enter a valid review.")

if __name__ == "__main__":
    main()
