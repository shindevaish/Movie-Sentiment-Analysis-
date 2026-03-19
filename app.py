import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

model = pickle.load(open('model/best_model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict sentiment (Positive / Negative)")

user_input = st.text_area("Enter Review")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review!")
    else:
        clean_text = preprocess(user_input)

        vector = vectorizer.transform([clean_text])

        prediction = model.predict(vector)[0]

        if prediction == 1 or prediction == "positive":
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")