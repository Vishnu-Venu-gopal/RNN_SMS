import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = tf.keras.models.load_model("https://github.com/Vishnu-Venu-gopal/RNN_SMS/blob/main/my_model.keras")

# Tokenizer and preprocessing functions
@st.cache_resource
def get_tokenizer():
    # Load and preprocess the dataset to create a tokenizer
    dataset = pd.read_csv('https://raw.githubusercontent.com/adityaiiitmk/Datasets/master/SMSSpamCollection', sep='\t', names=['label', 'message'])
    dataset['label'] = dataset['label'].map({'spam': 1, 'ham': 0})
    X_train, _, _, _ = train_test_split(dataset['message'].values, dataset['label'].values, test_size=0.3, random_state=42)
    tokeniser = tf.keras.preprocessing.text.Tokenizer()
    tokeniser.fit_on_texts(X_train)
    return tokeniser

# Preprocess the input text for prediction
def preprocess_text(text, tokenizer, max_length=10):
    encoded_text = tokenizer.texts_to_sequences([text])
    padded_text = tf.keras.preprocessing.sequence.pad_sequences(encoded_text, maxlen=max_length, padding='post')
    return padded_text

# Set up the Streamlit app
st.title("Spam Detection App")
st.write("This app uses a trained RNN model to classify SMS messages as **Spam** or **Ham**.")

# Load the tokenizer
tokenizer = get_tokenizer()

# Input text box for the user
user_input = st.text_input("Enter the message text here:")

# Button to trigger prediction
if st.button("Predict"):
    if user_input.strip() == "":
        st.write("Please enter a valid message.")
    else:
        # Preprocess the input and predict using the trained model
        processed_input = preprocess_text(user_input, tokenizer)
        prediction = (model.predict(processed_input) > 0.5).astype("int32")

        # Show the result
        if prediction[0][0] == 1:
            st.error("The message is classified as **Spam**.")
        else:
            st.success("The message is classified as **Ham**.")
