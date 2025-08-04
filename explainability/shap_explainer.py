import shap
import numpy as np

def explain_with_shap(model, resume_texts, jd_text):
    # SHAP expects a list of strings for both background and input
    def predict_fn(texts):
        embeddings = model.encode(texts)
        return np.array(embeddings)

    # Background data must be a list of strings
    background = resume_texts if isinstance(resume_texts, list) else [resume_texts]
    input_text = [jd_text] if isinstance(jd_text, str) else jd_text

    explainer = shap.Explainer(predict_fn, background)
    shap_values = explainer(input_text)
    return shap_values, explainer