import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

careers = pd.read_csv("careers.csv")

vectorizer = CountVectorizer()

skill_matrix = vectorizer.fit_transform(careers["skills"])

user_skills = input("Enter your skills: ")

all_skills = careers["skills"].tolist()

all_skills.append(user_skills)

matrix = vectorizer.fit_transform(all_skills)

similarity = cosine_similarity(matrix)

user_similarity = similarity[-1][:-1]

scores = list(enumerate(user_similarity))

scores = sorted(scores, key=lambda x: x[1], reverse=True)

print("\n===== Recommended Careers =====\n")

rank = 1

for career in scores[:5]:

    index = career[0]

    print(f"{rank}. {careers.iloc[index].career}")

    print(f"   Skills: {careers.iloc[index].skills}")

    print()

    rank += 1