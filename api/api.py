from flask import Flask, request, jsonify
from flask_cors import CORS  
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

model_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'

model = hub.load(model_url)

books = pd.read_csv('books.csv')

descriptions = books['treatment_1'].to_list()
descriptions_emb = model(descriptions)

@app.route("/prever", methods=["POST"])
def prever():
    dados = request.json 
    sentence = dados["entrada"]

    sentence_emb = model([sentence])
    similarities = cosine_similarity(sentence_emb, descriptions_emb).flatten()

    idxs = np.argsort(-similarities)

    recomendations = []

    for idx in idxs[:5]:
        book = books.iloc[idx]

        information = {
            "title": book['title'],
            "Author(s)": book['authors'],
            "Description": book['description'],
            "Category": book["categories"],
            "Average rating": book['average_rating'],
            "image_url": book["thumbnail"]
        }

        recomendations.append(information)

    return jsonify({"recomendations:": recomendations})

if __name__ == "__main__":
    app.run(debug=True)