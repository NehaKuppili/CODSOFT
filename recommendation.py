import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

careers = pd.read_csv("careers.csv")

with st.sidebar:

    st.title("About")

    st.write(
        "AI-powered career recommendation platform that suggests career paths based on user skills."
    )

    st.write("👩‍💻 Developed by Neha Kuppili")

    st.write("🎓 AIML Student")

    st.write("🚀 Career Guidance Tool")

st.title("🎯 AI Career Path Navigator")

st.write(
    "Discover career paths based on your skills using AI-powered matching and recommendation."
)

st.info("""
Popular Skills:
Python • SQL • Machine Learning • Deep Learning • React • AWS • Docker • Linux • TensorFlow • Power BI
""") 

user_skills = st.text_input(
    "Enter Skills",
    placeholder="python machine-learning sql"
)

if st.button("Recommend Careers"):

    vectorizer = CountVectorizer()

    all_skills = careers["skills"].tolist()

    all_skills.append(user_skills)

    matrix = vectorizer.fit_transform(all_skills)

    similarity = cosine_similarity(matrix)

    user_similarity = similarity[-1][:-1]

    scores = list(enumerate(user_similarity))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    st.subheader("Recommended Careers")

    for career in scores[:5]:

        index = career[0]

        score = round(career[1] * 100, 2)

        st.success(
            f"{careers.iloc[index].career} ({score}% Match)"
        )

        st.caption(
            f"Skills: {careers.iloc[index].skills}"
        )