import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# NLTK డేటా లోడ్ చేయడం
nltk.download('stopwords')
ps = PorterStemmer()

# 1. మోడల్ మరియు వెక్టరైజర్ ఫైల్స్ లోడ్ చేయడం
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# 2. టెక్స్ట్ క్లీనింగ్ ఫంక్షన్
def stemming(content):
    cleaned_content = re.sub('[^a-zA-Z]', ' ', content)
    cleaned_content = cleaned_content.lower()
    cleaned_content = cleaned_content.split()
    cleaned_content = [ps.stem(word) for word in cleaned_content if not word in stopwords.words('english')]
    return ' '.join(cleaned_content)

# 3. వెబ్‌సైట్ UI థీమ్ & డిజైన్ (Streamlit)
st.set_page_config(page_title="Fake News AI Detector", page_icon="🔍", layout="centered")

# కలర్‌ఫుల్ బ్యానర్ మరియు హెడర్ డిజైన్
st.markdown(
    """
    <div style="background-gradient: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                background-color: #1e3c72; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 25px;">
        <h1 style="color: white; margin-bottom: 5px; font-family: 'Helvetica Neue', sans-serif;">🔍 Smart Fake News Detector</h1>
        <p style="color: #e0e0e0; font-size: 16px; margin: 0;">Empowered by Machine Learning to verify news authenticity instantly</p>
    </div>
    """, 
    unsafe_allow_html=True
)

st.write("👋 **Welcome!** Copy any news paragraph or article from the web and paste it below to analyze its reliability.")

# యూజర్ టెక్స్ట్ ఎంటర్ చేయడానికి అందమైన బాక్స్
news_input = st.text_area(
    "📄 Enter the News Content Here:", 
    height=220, 
    placeholder="Type or paste the news article text here to check if it's real..."
)

st.markdown("<br>", unsafe_allow_html=True)

# బటన్ డిజైన్
if st.button("🚀 Analyze News Authenticity", use_container_width=True):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some text first to analyze!")
    else:
        with st.spinner("🕵️‍♂️ AI is scanning the text and patterns... Please wait..."):
            # 4. టెక్స్ట్‌ని క్లీన్ చేసి ప్రెడిక్ట్ చేయడం
            cleaned_news = stemming(news_input)
            vectorized_news = vectorizer.transform([cleaned_news])
            prediction = model.predict(vectorized_news)
            
            st.markdown("### 📊 Verification Result:")
            
            # 5. రిజల్ట్ చూపించడం (1 అంటే Fake, 0 అంటే Real)
            if prediction[0] == 1:
                st.markdown(
                    """
                    <div style="background-color: #ffdde1; border-left: 6px solid #d9534f; padding: 15px; border-radius: 8px;">
                        <h4 style="color: #d9534f; margin: 0;">🚨 WARNING: This news is likely FAKE!</h4>
                        <p style="color: #666; margin: 5px 0 0 0; font-size: 14px;">The language patterns in this text match common misinformation profiles.</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="background-color: #d4edda; border-left: 6px solid #28a745; padding: 15px; border-radius: 8px;">
                        <h4 style="color: #28a745; margin: 0;">✅ RELIABLE: This news is likely REAL!</h4>
                        <p style="color: #666; margin: 5px 0 0 0; font-size: 14px;">The system found this text to be consistent with trusted, verified news reporting.</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 13px;">
        Built with 💙 using Advanced Machine Learning & Streamlit UI
    </div>
    """, 
    unsafe_allow_html=True
)