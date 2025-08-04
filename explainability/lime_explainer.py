from lime.lime_text import LimeTextExplainer
import numpy as np

def explain_with_lime(model, resume_texts, jd_text):
    explainer = LimeTextExplainer(class_names=["match"])

    def predict_fn(texts):
        embeddings = model.encode(texts)
        # Convert to 2D array for LIME
        return np.array(embeddings)

    explanation = explainer.explain_instance(jd_text, predict_fn, num_features=10)
    return explanation