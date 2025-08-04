import shap

def explain_regressor(model, regressor, jd_text, background_texts):
    def predict_fn(texts):
        embeddings = model.encode(texts)
        return regressor.predict(embeddings)

    # Ensure inputs are list-like
    if isinstance(background_texts, str):
        background_texts = [background_texts]
    if isinstance(jd_text, str):
        jd_text = [jd_text]

    # ✅ Use SHAP Text masker
    masker = shap.maskers.Text()
    explainer = shap.Explainer(predict_fn, masker)

    shap_values = explainer(jd_text)
    return shap_values