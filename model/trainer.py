import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

def train_regressor(data_path="data/training_data.json"):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    with open(data_path, "r") as f:
        data = json.load(f)

    jd_texts = [item["jd"] for item in data]
    scores = [item["score"] for item in data]

    jd_embeddings = model.encode(jd_texts)
    X_train, X_test, y_train, y_test = train_test_split(jd_embeddings, scores, test_size=0.2)

    reg = Ridge()
    reg.fit(X_train, y_train)

    return model, reg