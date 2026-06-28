import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Load Dataset
# ----------------------------
careers = pd.read_csv("careers.csv")

# Remove empty rows
careers = careers.dropna(subset=["career", "skills"])

# Convert to string
careers["career"] = careers["career"].astype(str)
careers["skills"] = careers["skills"].astype(str)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.title("About")

    st.write(
        "AI-powered career recommendation platform that suggests career paths based on user skills."
    )

    st.write("👩‍💻 Developed by Neha Kuppili")
    st.write("🎓 AIML Student")
    st.write("🚀 Career Guidance Tool")

# ----------------------------
# Main Page
# ----------------------------
st.title("🎯 AI Career Path Navigator")

st.write(
    "Discover career paths based on your skills using AI-powered matching."
)

st.info("""
**Popular Skills**

Python • SQL • Machine Learning • Deep Learning • React • AWS • Docker • Linux • TensorFlow • Power BI
""")

user_skills = st.text_input(
    "Enter your skills",
    placeholder="python sql machine learning"
)

if st.button("Recommend Careers"):

    if user_skills.strip() == "":
        st.warning("Please enter your skills.")
    else:

        vectorizer = CountVectorizer()

        all_skills = careers["skills"].tolist()
        all_skills.append(user_skills.lower())

        matrix = vectorizer.fit_transform(all_skills)

        similarity = cosine_similarity(matrix)

        user_similarity = similarity[-1][:-1]

        scores = list(enumerate(user_similarity))

        scores = sorted(
            scores,
            key=lambda x: x[1],
            reverse=True
        )

        st.subheader("🎯 Top Career Recommendations")

        found = False

        for index, score in scores[:5]:

            if score > 0:

                found = True

                st.success(
                    f"✅ {careers.iloc[index]['career']} ({score*100:.2f}% Match)"
                )

                st.write(
                    f"**Required Skills:** {careers.iloc[index]['skills']}"
                )

                st.divider()

        if not found:
            st.error(
                "No suitable career found. Try entering more relevant skills."
            )
            