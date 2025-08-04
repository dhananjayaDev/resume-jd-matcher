# import streamlit as st
# from sentence_transformers import SentenceTransformer
# from model.trainer import train_regressor
# from explainability.shap_regressor import explain_regressor
# from explainability.lime_explainer import explain_with_lime
# from utils.preprocess import clean_text
# import streamlit.components.v1 as components
# import shap

# # Page config
# st.set_page_config(page_title="Resume & JD Matcher", layout="wide")

# # Helper: Style SHAP HTML for better visibility
# def styled_shap_html(html_plot):
#     styled_html = f"""
#     <style>
#         .shap-plot * {{
#             background-color: #f9f9f9 !important;
#             color: #222 !important;
#             font-family: 'Segoe UI', sans-serif;
#         }}
#     </style>
#     <div class="shap-plot">{html_plot}</div>
#     """
#     return styled_html

# # Centered layout using columns
# col1, col2, col3 = st.columns([1, 2, 1])
# with col2:
#     st.title("Resume & JD Matcher with Explainability")
#     st.markdown("---")

#     # Input fields
#     st.header("Input")
#     resume = st.text_area("Paste Resume", height=150)
#     jd = st.text_area("Paste Job Description", height=150)

#     if st.button("Match & Explain"):
#         st.markdown("---")

#         # Clean inputs
#         resume_clean = clean_text(resume)
#         jd_clean = clean_text(jd)

#         # Load model + regressor
#         model, regressor = train_regressor()

#         # Predict match score
#         jd_emb = model.encode(jd_clean)
#         score = regressor.predict([jd_emb])[0]
#         st.metric(label="Predicted Match Score", value=round(score, 3))

#         # SHAP Explanation
#         st.header("SHAP Explanation")
#         try:
#             shap_values = explain_regressor(model, regressor, jd_clean, [resume_clean])
#             st.markdown("**SHAP Values**")
#             st.write(shap_values.values)
#             st.markdown("**Base Value**")
#             st.write(shap_values.base_values)

#             # SHAP Text Visualization
#             try:
#                 html_plot = shap.plots.text(shap_values[0], display=False)
#                 styled_plot = styled_shap_html(html_plot)
#                 components.html(styled_plot, height=300, scrolling=True)
#             except Exception as e:
#                 st.warning(f"SHAP plot failed: {e}")

#         except Exception as e:
#             st.warning(f"SHAP explanation failed: {e}")

#         # LIME Explanation
#         st.header("LIME Explanation")
#         try:
#             lime_exp = explain_with_lime(model, [resume_clean], jd_clean)
#             st.markdown("**LIME Contributions**")
#             for word, weight in lime_exp.as_list():
#                 st.markdown(
#                     f"<div style='padding-left:10px'><strong>{word}</strong>: {round(weight, 4)}</div>",
#                     unsafe_allow_html=True
#                 )
#         except Exception as e:
#             st.warning(f"LIME explanation failed: {e}")

import streamlit as st
from sentence_transformers import SentenceTransformer
from model.trainer import train_regressor
from explainability.shap_regressor import explain_regressor
from explainability.lime_explainer import explain_with_lime
from utils.preprocess import clean_text
import streamlit.components.v1 as components
import shap

# Page config
st.set_page_config(page_title="Resume & JD Matcher", layout="wide")

# Helper: Style SHAP HTML for better visibility and scrolling
def styled_shap_html(html_plot):
    styled_html = f"""
    <style>
        .scroll-box {{
            overflow-x: auto;
            overflow-y: auto;
            max-height: 300px;
            max-width: 100%;
            border: 1px solid #ccc;
            padding: 10px;
            background-color: #f9f9f9;
        }}
        .shap-plot * {{
            font-family: 'Segoe UI', sans-serif;
            font-size: 14px;
            color: #222 !important;
        }}
    </style>
    <div class="scroll-box shap-plot">{html_plot}</div>
    """
    return styled_html

# Title
st.markdown("<h2 style='text-align: center; margin-bottom: 10px;'>Resume & JD Matcher Dashboard</h2>", unsafe_allow_html=True)

# Compact layout: Inputs (left) and Results (right)
input_col, result_col = st.columns([1, 1.5])

with input_col:
    st.markdown("#### Input")
    resume = st.text_area("Resume", height=120)
    jd = st.text_area("Job Description", height=120)
    match_triggered = st.button("Match & Explain")

with result_col:
    if match_triggered:
        resume_clean = clean_text(resume)
        jd_clean = clean_text(jd)

        model, regressor = train_regressor()
        jd_emb = model.encode(jd_clean)
        score = regressor.predict([jd_emb])[0]

        st.markdown("#### Match Score")
        st.metric(label="Score", value=round(score, 3))

        # SHAP Explanation
        st.markdown("#### SHAP Explanation")
        try:
            shap_values = explain_regressor(model, regressor, jd_clean, [resume_clean])

            # Show SHAP values and base value
            st.markdown("**SHAP Values:**")
            st.write(shap_values.values)
            st.markdown("**Base Value:**")
            st.write(shap_values.base_values)

            # SHAP Text Visualization
            try:
                html_plot = shap.plots.text(shap_values[0], display=False)
                styled_plot = styled_shap_html(html_plot)
                components.html(styled_plot, height=320, scrolling=False)
            except Exception as e:
                st.warning(f"SHAP plot failed: {e}")

        except Exception as e:
            st.warning(f"SHAP explanation failed: {e}")

        # LIME Explanation
        st.markdown("#### LIME Explanation")
        try:
            lime_exp = explain_with_lime(model, [resume_clean], jd_clean)
            lime_html = "<div style='font-size:14px; padding-left:8px;'>"
            for word, weight in lime_exp.as_list():
                lime_html += f"<div><strong>{word}</strong>: {round(weight, 4)}</div>"
            lime_html += "</div>"
            st.markdown(lime_html, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"LIME explanation failed: {e}")