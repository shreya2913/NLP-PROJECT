import streamlit as st  
import pandas as pd  
from transformers import AutoTokenizer, AutoModelForSequenceClassification  
import torch

# Load the BERT model and tokenizer  
model_name = "bert-base-uncased"  
tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load the data from the CSV file  
df = pd.read_csv("data.csv")

# Ensure the CSV has the required columns
if 'reviews' not in df.columns or 'reviews_sentiment' not in df.columns:
    st.error("CSV must contain 'reviews' and 'reviews_sentiment' columns.")
else:
    st.write("CSV loaded successfully!")

# Define the sentiment analysis function
def predict_sentiment(review):
    inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return "POSITIVE" if prediction == 1 else "NEGATIVE"

# Define the Streamlit app
def main():
    st.title("BERT TEXT CLASSIFICATION")  # Updated title

    # Display the data from the CSV file  
    st.subheader("Data from CSV file")  
    st.write(df)  # Display the entire DataFrame

    # Get the user's input  
    review = st.text_area("Enter a shoe review:")

    # Perform sentiment analysis on user input
    if review:
        if st.button("Predict Sentiment"):
            sentiment = predict_sentiment(review)
            st.write(f"**Predicted Sentiment:** {sentiment}")

if __name__ == "__main__":  
    main()
