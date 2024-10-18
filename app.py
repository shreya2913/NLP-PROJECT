import streamlit as st
import pandas as pd
import torch
import pickle

# Set Streamlit page configuration
st.set_page_config(page_title="BERT Sentiment Analysis", layout="wide")

# Load the BERT model, tokenizer, and label encoder from the pickle file
@st.cache_resource  # Caching to avoid reloading the model on each run
def load_model():
    with open('BERT.pkl', 'rb') as model_file:
        tokenizer, data, model, label_encoder = pickle.load(model_file)
    return tokenizer, model, label_encoder

tokenizer, model, label_encoder = load_model()

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
    
    return decoded_label, probabilities.cpu().numpy()[0]

# Streamlit app
def main():
    st.title("BERT Sentiment Analysis for Reviews")
    st.subheader("Enter a Review for Sentiment Prediction")

    # User input
    user_input = st.text_area("Enter your review:", height=150, placeholder="Type your text here...")

    if st.button("Predict Sentiment"):
        if user_input.strip():
            with st.spinner('Predicting sentiment...'):
                try:
                    # Make the prediction
                    predicted_label, probabilities = predict(user_input, tokenizer, model, label_encoder)

                    # Display predicted label
                    st.success(f"**Predicted Sentiment:** {predicted_label}")

                    # Display prediction probabilities as a DataFrame
                    labels = label_encoder.classes_
                    prediction_df = pd.DataFrame([probabilities], columns=labels)
                    st.write("**Prediction Probabilities:**")
                    st.dataframe(prediction_df)

                    # Display probabilities as a bar chart
                    st.bar_chart(prediction_df.T)

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Please enter some text for prediction.")

if __name__ == "__main__":
    main()
