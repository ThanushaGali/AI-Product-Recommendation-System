from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load product data
try:
    df = pd.read_csv("product.csv")
except FileNotFoundError:
    df = pd.DataFrame(columns=["product_id", "product_name", "category", "price", "rating", "reviews"])
    print("⚠️ Warning: product.csv not found! Ensure it exists in the correct folder.")

# Get unique categories
categories = df["category"].unique() if not df.empty else []

# AI-Based Recommendations using NLP
def recommend_by_description(keyword, top_n=5):
    if "reviews" not in df.columns or df["reviews"].isnull().all():
        return []  # Return empty if descriptions/reviews are missing

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["reviews"].fillna(""))

    # Transform the keyword input into the same space as the product reviews
    keyword_vector = vectorizer.transform([keyword])

    # Compute cosine similarity
    similarity_scores = cosine_similarity(keyword_vector, tfidf_matrix).flatten()

    # Get top N similar products
    top_indices = similarity_scores.argsort()[-top_n:][::-1]

    return df.iloc[top_indices][["product_name", "category", "price", "rating", "reviews"]].to_dict(orient="records")

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []

    if request.method == "POST":
        category = request.form.get("category", "").strip()
        keyword = request.form.get("keyword", "").strip()
        min_rating = float(request.form.get("min_rating", "0") or "0")
        max_price = float(request.form.get("max_price", "99999") or "99999")

        if category:
            recommendations = df[
                (df["category"].str.lower() == category.lower()) &
                (df["rating"] >= min_rating) &
                (df["price"] <= max_price)
            ][["product_name", "category", "price", "rating", "reviews"]].to_dict(orient="records")

        elif keyword:
            recommendations = recommend_by_description(keyword)

    return render_template("index.html", recommendations=recommendations, categories=categories)

if __name__ == "__main__":
    app.run(debug=True)
