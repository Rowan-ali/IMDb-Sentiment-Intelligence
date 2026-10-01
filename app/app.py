# ============================================================
# IMDb SENTIMENT INTELLIGENCE
# Professional Streamlit Deployment
# ============================================================

import os
import re
import joblib
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IMDb Sentiment Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True


# ============================================================
# SIDEBAR — THEME CONTROL
# ============================================================

with st.sidebar:

    st.markdown("## 🎬 Sentiment Intelligence")

    st.caption(
        "Professional NLP sentiment classification interface"
    )

    st.divider()

    dark_mode = st.toggle(
        "🌙 Dark Mode",
        value=st.session_state.dark_mode
    )

    st.session_state.dark_mode = dark_mode


# ============================================================
# THEME COLORS
# ============================================================

if dark_mode:

    PAGE_BG = "#0B1120"
    SECONDARY_BG = "#111827"
    CARD_BG = "#151F32"

    TEXT_PRIMARY = "#F8FAFC"
    TEXT_SECONDARY = "#94A3B8"

    BORDER = "#263449"

    INPUT_BG = "#111827"

    POSITIVE_BG = "#0D2D24"
    POSITIVE_TEXT = "#4ADE80"

    NEGATIVE_BG = "#35171D"
    NEGATIVE_TEXT = "#FB7185"

    ACCENT = "#818CF8"

else:

    PAGE_BG = "#F6F8FC"
    SECONDARY_BG = "#FFFFFF"
    CARD_BG = "#FFFFFF"

    TEXT_PRIMARY = "#172033"
    TEXT_SECONDARY = "#64748B"

    BORDER = "#E2E8F0"

    INPUT_BG = "#FFFFFF"

    POSITIVE_BG = "#ECFDF5"
    POSITIVE_TEXT = "#15803D"

    NEGATIVE_BG = "#FFF1F2"
    NEGATIVE_TEXT = "#BE123C"

    ACCENT = "#4F46E5"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    f"""
<style>

.stApp {{
    background: {PAGE_BG};
    color: {TEXT_PRIMARY};
}}

.block-container {{
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}}

[data-testid="stSidebar"] {{
    background: {SECONDARY_BG};
    border-right: 1px solid {BORDER};
}}

[data-testid="stSidebar"] * {{
    color: {TEXT_PRIMARY};
}}

h1, h2, h3, h4 {{
    color: {TEXT_PRIMARY} !important;
}}

p, label {{
    color: {TEXT_SECONDARY};
}}


/* -------------------------------
   HEADER
-------------------------------- */

.main-header {{
    padding: 12px 2px 22px 2px;
}}

.app-badge {{
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(99, 102, 241, 0.12);
    color: {ACCENT};
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 12px;
}}

.app-title {{
    font-size: 44px;
    font-weight: 800;
    line-height: 1.1;
    color: {TEXT_PRIMARY};
    margin: 0;
}}

.app-subtitle {{
    margin-top: 10px;
    max-width: 800px;
    font-size: 17px;
    line-height: 1.7;
    color: {TEXT_SECONDARY};
}}


/* -------------------------------
   METRIC CARDS
-------------------------------- */

.metric-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 20px 22px;
    min-height: 115px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.04);
}}

.metric-label {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}}

.metric-value {{
    margin-top: 9px;
    color: {TEXT_PRIMARY};
    font-size: 22px;
    font-weight: 750;
}}


/* -------------------------------
   ANALYSIS PANEL
-------------------------------- */

.analysis-panel {{
    margin-top: 30px;
    margin-bottom: 14px;
    padding: 24px;
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 18px;
}}

.panel-title {{
    font-size: 24px;
    font-weight: 750;
    color: {TEXT_PRIMARY};
}}

.panel-description {{
    color: {TEXT_SECONDARY};
    margin-top: 6px;
    line-height: 1.6;
}}


/* -------------------------------
   INPUTS
-------------------------------- */

.stTextArea textarea {{
    background: {INPUT_BG} !important;
    color: {TEXT_PRIMARY} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
}}

.stSelectbox div[data-baseweb="select"] > div {{
    background: {INPUT_BG};
    color: {TEXT_PRIMARY};
    border-color: {BORDER};
}}


/* -------------------------------
   BUTTON
-------------------------------- */

.stButton > button {{
    border-radius: 12px;
    min-height: 48px;
    font-size: 16px;
    font-weight: 700;
}}


/* -------------------------------
   RESULT
-------------------------------- */

.positive-result {{
    background: {POSITIVE_BG};
    border: 1px solid {POSITIVE_TEXT};
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    margin-top: 20px;
}}

.negative-result {{
    background: {NEGATIVE_BG};
    border: 1px solid {NEGATIVE_TEXT};
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    margin-top: 20px;
}}

.positive-result-title {{
    color: {POSITIVE_TEXT};
    font-size: 32px;
    font-weight: 800;
}}

.negative-result-title {{
    color: {NEGATIVE_TEXT};
    font-size: 32px;
    font-weight: 800;
}}

.result-confidence {{
    color: {TEXT_SECONDARY};
    font-size: 15px;
    margin-top: 7px;
}}


/* -------------------------------
   FOOTER
-------------------------------- */

.footer {{
    text-align: center;
    color: {TEXT_SECONDARY};
    margin-top: 40px;
    font-size: 13px;
}}

footer {{
    visibility: hidden;
}}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_review(text):

    # Convert to string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove Keras special tokens
    text = re.sub(
        r"<(?:start|unk|pad|unused)>",
        " ",
        text
    )

    # Remove HTML line breaks
    text = re.sub(
        r"<br\s*/?>",
        " ",
        text
    )

    # Remove reconstructed IMDb 'br' tokens
    text = re.sub(
        r"\bbr\b",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Keep alphabetic characters and apostrophes
    text = re.sub(
        r"[^a-zA-Z\s']",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "sentiment_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "tfidf_vectorizer.pkl"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_resources():

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            "sentiment_model.pkl was not found."
        )

    if not os.path.isfile(VECTORIZER_PATH):
        raise FileNotFoundError(
            "tfidf_vectorizer.pkl was not found."
        )

    loaded_model = joblib.load(
        MODEL_PATH
    )

    loaded_vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    return loaded_model, loaded_vectorizer


try:

    model, vectorizer = load_resources()

except Exception as error:

    st.error(
        "Model resources could not be loaded."
    )

    st.code(
        str(error)
    )

    st.info(
        "Keep app.py, sentiment_model.pkl and "
        "tfidf_vectorizer.pkl in the same folder."
    )

    st.stop()


# ============================================================
# SIDEBAR INFORMATION
# ============================================================

with st.sidebar:

    st.divider()

    st.markdown("### Model")

    st.write(
        "**Optimized TF-IDF + Logistic Regression**"
    )

    st.caption(
        "Selected as the strongest model evaluated "
        "on the complete official test set."
    )

    st.divider()

    st.markdown("### Test Performance")

    col_a, col_b = st.columns(2)

    with col_a:
        st.metric(
            "Accuracy",
            "89.58%"
        )

    with col_b:
        st.metric(
            "F1",
            "89.62%"
        )

    col_c, col_d = st.columns(2)

    with col_c:
        st.metric(
            "Precision",
            "89.31%"
        )

    with col_d:
        st.metric(
            "Recall",
            "89.94%"
        )

    st.divider()

    st.caption(
        "Dataset: IMDb Movie Reviews"
    )

    st.caption(
        "Evaluation set: 25,000 reviews"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
<div class="main-header">
<span class="app-badge">NATURAL LANGUAGE PROCESSING</span>
<div class="app-title">🎬 IMDb Sentiment Intelligence</div>
<div class="app-subtitle">
An interactive sentiment-analysis system that classifies movie
reviews using the optimized TF-IDF representation and Logistic
Regression model developed during the NLP project.
</div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# MODEL OVERVIEW
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
<div class="metric-card">
<div class="metric-label">Text Representation</div>
<div class="metric-value">TF-IDF</div>
</div>
""",
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
<div class="metric-card">
<div class="metric-label">Classifier</div>
<div class="metric-value">Logistic Regression</div>
</div>
""",
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
<div class="metric-card">
<div class="metric-label">Official Test Accuracy</div>
<div class="metric-value">89.58%</div>
</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# ANALYSIS HEADER
# ============================================================

st.markdown(
    """
<div class="analysis-panel">
<div class="panel-title">Analyze a Movie Review</div>
<div class="panel-description">
Enter a review below and the NLP pipeline will clean the text,
generate TF-IDF features, and classify its overall sentiment.
</div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# EXAMPLE REVIEWS
# ============================================================

positive_example = (
    "This movie was absolutely fantastic. "
    "The performances were excellent, the story was engaging, "
    "and I enjoyed every minute of it."
)

negative_example = (
    "This movie was extremely disappointing. "
    "The story was boring, the acting was weak, "
    "and I would not recommend it."
)


example_choice = st.selectbox(
    "Example review",
    options=[
        "Write my own review",
        "Load positive example",
        "Load negative example"
    ]
)


if example_choice == "Load positive example":

    default_review = positive_example

elif example_choice == "Load negative example":

    default_review = negative_example

else:

    default_review = ""


# ============================================================
# TEXT INPUT
# ============================================================

review = st.text_area(
    "Movie review",
    value=default_review,
    placeholder=(
        "Write or paste an IMDb-style movie review here..."
    ),
    height=190
)


# ============================================================
# INPUT STATISTICS
# ============================================================

word_count = (
    len(review.split())
    if review.strip()
    else 0
)

character_count = len(review)


stats_col1, stats_col2, stats_col3 = st.columns(3)

with stats_col1:

    st.caption(
        f"📝 Words: {word_count:,}"
    )

with stats_col2:

    st.caption(
        f"🔤 Characters: {character_count:,}"
    )

with stats_col3:

    st.caption(
        "⚙️ Model ready"
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "Analyze Sentiment",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    if not review.strip():

        st.warning(
            "Please enter a movie review first."
        )

    else:

        cleaned_review = clean_review(
            review
        )

        if not cleaned_review:

            st.warning(
                "The entered review does not contain "
                "enough valid textual information."
            )

        else:

            # Transform with trained TF-IDF vectorizer
            review_vector = vectorizer.transform(
                [cleaned_review]
            )

            # Class prediction
            prediction = int(
                model.predict(
                    review_vector
                )[0]
            )

            # Class probabilities
            probabilities = model.predict_proba(
                review_vector
            )[0]

            negative_probability = float(
                probabilities[0]
            )

            positive_probability = float(
                probabilities[1]
            )


            # ====================================================
            # RESULT VALUES
            # ====================================================

            if prediction == 1:

                predicted_label = "POSITIVE"

                confidence = (
                    positive_probability
                )

                result_html = f"""
<div class="positive-result">
<div class="positive-result-title">
😊 POSITIVE SENTIMENT
</div>
<div class="result-confidence">
Model confidence:
<strong>{confidence * 100:.2f}%</strong>
</div>
</div>
"""

            else:

                predicted_label = "NEGATIVE"

                confidence = (
                    negative_probability
                )

                result_html = f"""
<div class="negative-result">
<div class="negative-result-title">
😞 NEGATIVE SENTIMENT
</div>
<div class="result-confidence">
Model confidence:
<strong>{confidence * 100:.2f}%</strong>
</div>
</div>
"""


            # ====================================================
            # DISPLAY RESULT
            # ====================================================

            st.markdown(
                "## Prediction Result"
            )

            st.markdown(
                result_html,
                unsafe_allow_html=True
            )


            # ====================================================
            # PROBABILITY METRICS
            # ====================================================

            st.markdown(
                "### Sentiment Probability"
            )

            positive_col, negative_col = st.columns(2)

            with positive_col:

                st.metric(
                    "😊 Positive",
                    f"{positive_probability * 100:.2f}%"
                )

                st.progress(
                    positive_probability
                )


            with negative_col:

                st.metric(
                    "😞 Negative",
                    f"{negative_probability * 100:.2f}%"
                )

                st.progress(
                    negative_probability
                )


            # ====================================================
            # CONFIDENCE LEVEL
            # ====================================================

            st.markdown(
                "### Prediction Confidence"
            )

            if confidence >= 0.90:

                st.success(
                    "High-confidence prediction"
                )

            elif confidence >= 0.70:

                st.info(
                    "Moderate-confidence prediction"
                )

            else:

                st.warning(
                    "Lower-confidence prediction — the review "
                    "may contain mixed or ambiguous sentiment."
                )


            # ====================================================
            # TECHNICAL DETAILS
            # ====================================================

            with st.expander(
                "View prediction details"
            ):

                st.write(
                    f"**Predicted Class:** {predicted_label}"
                )

                st.write(
                    f"**Positive Probability:** "
                    f"{positive_probability:.4f}"
                )

                st.write(
                    f"**Negative Probability:** "
                    f"{negative_probability:.4f}"
                )

                st.write(
                    f"**Model Confidence:** "
                    f"{confidence:.4f}"
                )


            with st.expander(
                "View processed review"
            ):

                st.write(
                    cleaned_review
                )


            with st.expander(
                "How does the prediction work?"
            ):

                st.markdown(
                    """
The application follows the same inference pipeline used
during project development:

**Review Input → Text Cleaning → TF-IDF Transformation → Logistic Regression → Sentiment Prediction**

The TF-IDF vectorizer and Logistic Regression classifier are
loaded from the trained project artifacts. The application
therefore performs inference only and does not retrain the
model during startup.
"""
                )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

with st.expander(
    "About this project"
):

    st.markdown(
        """
### NLP Sentiment Classification Project

Several Natural Language Processing and deep-learning
approaches were investigated during the project, including
TF-IDF, Word2Vec, Simple RNN, LSTM, Bidirectional LSTM,
GRU and a pretrained Transformer.

The optimized **TF-IDF + Logistic Regression** pipeline was
selected for this application because it achieved the
strongest directly comparable performance on the complete
**25,000-review official test set** while remaining highly
efficient for real-time deployment.

**Official test performance**

- Accuracy: **89.58%**
- Precision: **89.31%**
- Recall: **89.94%**
- F1-score: **89.62%**
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
IMDb Sentiment Intelligence · NLP Project<br>
Python · scikit-learn · Streamlit
</div>
""",
    unsafe_allow_html=True
)