import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data (first time)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Load model and vectorizer
model = pickle.load(open('model/best_model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

# Preprocessing setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Streamlit UI
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict sentiment (Positive / Negative)")

# Input box
user_input = st.text_area("Enter Review")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review!")
    else:
        # Preprocess
        clean_text = preprocess(user_input)

        # Vectorize
        vector = vectorizer.transform([clean_text])

        # Predict
        prediction = model.predict(vector)[0]

        # Output
        if prediction == 1 or prediction == "positive":
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")