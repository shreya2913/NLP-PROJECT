import streamlit as st
import pandas as pd
import torch
import pickle

# Load the BERT model, tokenizer, and label encoder from the pickle file
with open('BERT.pkl', 'rb') as model_file:
    tokenizer, data, model, label_encoder = pickle.load(model_file)

# Prediction function
def predict(text, tokenizer, model, label_encoder):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True, 
        padding="max_length"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    predicted_label = torch.argmax(probabilities, dim=1).item()
    decoded_label = label_encoder.inverse_transform([predicted_label])[0]
    
    return decoded_label, probabilities

# Streamlit app
def main():
    st.title("BERT Sentiment Analysis for Reviews")
    st.subheader("Enter a Review for Sentiment Prediction")

    # Text area for user input
    user_input = st.text_area("Enter your review:", height=150, placeholder="Type your text here...")

    if st.button("Predict Sentiment"):
        if user_input.strip():
            with st.spinner('Predicting sentiment...'):
                try:
                    # Make the prediction
                    predicted_label, probabilities = predict(user_input, tokenizer, model, label_encoder)

                    # Display the predicted label
                    st.write(f"**Predicted Sentiment:** {predicted_label}")

                    # Create a DataFrame for prediction probabilities
                    prediction_df = pd.DataFrame(
                        probabilities.numpy(), 
                        columns=["Negative", "Neutral", "Positive"]
                    )

                    # Display the probabilities
                    st.write("**Prediction Probabilities:**")
                    st.dataframe(prediction_df)

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Please enter some text for prediction.")

if __name__ == "__main__":
    main()
