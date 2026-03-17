import streamlit as st
import pickle
from pathlib import Path
import os

# Load model and vectorizer
BASE_DIR = Path(__file__).parent
model_path = BASE_DIR / "spam_model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)
model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

# Page config
st.set_page_config(
    page_title="Spam Classifier Demo",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for Dark Theme + Buttons + Cards
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #0d1117;
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar */
.css-1d391kg {
    background-color: #161b22;
}

/* Text area */
textarea, input {
    background-color: #161b22;
    color: #ffffff;
    border: 1px solid #30363d;
}

/* Gradient Buttons */
.stButton>button {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    font-weight: bold;
    border-radius: 8px;
    height: 3em;
}

/* Cards for results */
.card {
    padding: 20px;
    margin-top: 10px;
    border-radius: 12px;
    background-color: #21262d;
}
</style>
""", unsafe_allow_html=True)

# Sidebar info
st.sidebar.title("About")
st.sidebar.info("""
📌 **Spam Classifier Demo**  
Classifies messages as **Spam** or **Not Spam** using ML.  

- Model: Naive Bayes  
- Features: TF-IDF  
- Built with Python & Streamlit  
""")

# App Title
st.title("📩 Spam Message Classifier")
st.write("Enter a message or select an example to see predictions:")

# Example messages
examples = [
    "Congratulations! You won a free iPhone. Click here now!",
    "Hey, are we meeting tomorrow?",
    "URGENT: Your account will be suspended unless you act now!",
    "Don't forget to submit your assignment by 5 PM."
]

# Layout: input left, results right
col1, col2 = st.columns([2, 1])

with col1:
    message = st.text_area("Your message here:")

    selected_example = st.selectbox("Or try an example message:", ["--Select--"] + examples)
    if selected_example != "--Select--":
        message = selected_example

    if st.button("Predict"):
        if message.strip() == "":
            st.warning("Please enter a message!")
        else:
            data = vectorizer.transform([message])
            prediction = model.predict(data)[0]
            probability = model.predict_proba(data)[0][1]

            with col2:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="card">
                    <h3>🚨 Spam Message</h3>
                    <p>Confidence: {probability:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="card">
                    <h3>✅ Not Spam</h3>
                    <p>Confidence: {1 - probability:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 📝 Results will appear here")
