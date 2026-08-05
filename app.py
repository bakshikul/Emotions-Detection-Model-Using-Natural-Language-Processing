"""
Emotion Detection NLP App
-------------------------
Streamlit UI for TF-IDF + Logistic Regression emotion classifier.

Required files in the same folder:
    - app.py
    - NLP_Emotions_model.joblib
    - NLP_Emotions_vectorizer.joblib

The model and TF-IDF vectorizer must be the exact pair produced
during model training.
"""

import os
import string

import joblib
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "NLP_Emotions_model.joblib")
VECTORIZER_PATH = os.path.join(APP_DIR, "NLP_Emotions_vectorizer.joblib")

# IMPORTANT: this order must match the class order used during training.
EMOTION_ORDER = ["sadness", "anger", "love", "surprise", "fear", "joy"]

EMOTION_META = {
    "joy":      {"emoji": "😄", "color": "#FFC93C", "label": "Joy"},
    "sadness":  {"emoji": "😢", "color": "#4C6EF5", "label": "Sadness"},
    "anger":    {"emoji": "😠", "color": "#FF5C5C", "label": "Anger"},
    "fear":     {"emoji": "😨", "color": "#8854D0", "label": "Fear"},
    "love":     {"emoji": "❤️", "color": "#FF6FA5", "label": "Love"},
    "surprise": {"emoji": "😲", "color": "#2ED8B6", "label": "Surprise"},
}


# ============================================================
# FALLBACK STOPWORDS
# ============================================================

FALLBACK_STOPWORDS = set("""
i me my myself we our ours ourselves you you're you've you'll you'd your
yours yourself yourselves he him his himself she she's her hers herself it
it's its itself they them their theirs themselves what which who whom this
that that'll these those am is are was were be been being have has had
having do does did doing a an the and but if or because as until while of
at by for with about against between into through during before after
above below to from up down in out on off over under again further then
once here there when where why how all any both each few more most other
some such no nor not only own same so than too very s t can will just don
don't should should've now d ll m o re ve y ain aren aren't couldn couldn't
didn didn't doesn doesn't hadn hadn't hasn hasn't haven haven't isn isn't
ma mightn mightn't mustn mustn't needn needn't shan shan't shouldn
shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
""".split())


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def remove_punctuation(text):
    """Remove punctuation from text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text):
    """Remove numbers from text."""
    return "".join(ch for ch in text if not ch.isdigit())


@st.cache_resource(show_spinner=False)
def load_nltk_assets():
    """
    Load NLTK tokenizer and English stopwords.

    If NLTK resources are unavailable, a simple whitespace
    tokenizer and fallback stopword list are used instead.
    """
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)

        stop_words = set(stopwords.words("english"))
        return word_tokenize, stop_words

    except Exception:
        return (lambda text: text.split(), FALLBACK_STOPWORDS)


def clean_text(text, tokenizer, stop_words):
    """
    Apply the same basic preprocessing used during training:
    lowercase -> remove punctuation -> remove numbers ->
    tokenize -> remove stopwords.
    """
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_numbers(text)
    words = tokenizer(text)
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


# ============================================================
# LOAD MODEL + VECTORIZER
# ============================================================

@st.cache_resource(show_spinner=True)
def load_pipeline():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}\n\n"
            "Make sure NLP_Emotions_model.joblib is in "
            "the same folder as app.py."
        )

    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Vectorizer file not found:\n{VECTORIZER_PATH}\n\n"
            "Make sure NLP_Emotions_vectorizer.joblib is "
            "in the same folder as app.py."
        )

    tokenizer, stop_words = load_nltk_assets()
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    model_features = getattr(model, "n_features_in_", None)
    vectorizer_features = len(vectorizer.vocabulary_)

    if model_features is not None and model_features != vectorizer_features:
        raise ValueError(
            "\n\nFEATURE MISMATCH!\n\n"
            f"Model expects {model_features} features.\n"
            f"Vectorizer produces {vectorizer_features} features.\n\n"
            "This means the model and vectorizer were not "
            "created from the same training run.\n\n"
            "Retrain the model and save both files together."
        )

    return {
        "tokenizer": tokenizer,
        "stop_words": stop_words,
        "vectorizer": vectorizer,
        "model": model,
    }


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_emotion(text, pipeline):
    cleaned_text = clean_text(text, pipeline["tokenizer"], pipeline["stop_words"])

    # IMPORTANT: use transform(), never fit_transform(), on the saved vectorizer.
    vector = pipeline["vectorizer"].transform([cleaned_text])
    probabilities = pipeline["model"].predict_proba(vector)[0]

    return {
        emotion: float(probabilities[i])
        for i, emotion in enumerate(EMOTION_ORDER)
    }


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS — modernized theme
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(76,110,245,0.18) 0%, transparent 40%),
            radial-gradient(circle at 85% 0%, rgba(255,111,165,0.14) 0%, transparent 45%),
            linear-gradient(180deg, #12142b 0%, #0b0c18 100%);
        color: #e9eaf2;
    }

    #MainMenu, footer, header {visibility: hidden;}

    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.03);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* ---------- Hero ---------- */
    .hero {
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
    }

    .hero .badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #b9c0ff;
        background: rgba(76,110,245,0.14);
        border: 1px solid rgba(76,110,245,0.35);
        margin-bottom: 0.9rem;
    }

    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #ff6fa5, #4c6ef5 55%, #2ed8b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.35rem 0;
    }

    .hero p {
        color: #9aa0b4;
        font-size: 1.02rem;
        margin: 0;
    }

    /* ---------- Card container ---------- */
    .panel {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.4rem 1.4rem 1.1rem 1.4rem;
        margin-bottom: 1.1rem;
        backdrop-filter: blur(6px);
    }

    .panel-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #cdd1e8;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* ---------- Text area ---------- */
    .stTextArea textarea {
        border-radius: 14px !important;
        font-size: 1rem !important;
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        color: #f1f2f8 !important;
    }

    .stTextArea textarea:focus {
        border: 1px solid #4c6ef5 !important;
        box-shadow: 0 0 0 3px rgba(76,110,245,0.25) !important;
    }

    /* ---------- Example chips ---------- */
    div.stButton > button {
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.85rem;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.04);
        color: #d6d9ea;
        transition: all 0.15s ease;
    }

    div.stButton > button:hover {
        border-color: #4c6ef5;
        color: #ffffff;
        background: rgba(76,110,245,0.16);
    }

    /* ---------- Primary predict button ---------- */
    div[data-testid="stButton"]:has(button:contains("Predict")) button {
        background: linear-gradient(90deg, #4c6ef5, #ff6fa5);
    }

    .predict-btn button {
        border-radius: 999px !important;
        padding: 0.7rem 1.6rem !important;
        font-weight: 700 !important;
        font-size: 1.02rem !important;
        border: none !important;
        background: linear-gradient(90deg, #4c6ef5, #ff6fa5) !important;
        color: white !important;
        box-shadow: 0 6px 22px rgba(76,110,245,0.35);
    }

    .predict-btn button:hover {
        opacity: 0.92;
        color: white !important;
        transform: translateY(-1px);
    }

    /* ---------- Result card ---------- */
    .result-card {
        border-radius: 22px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin-top: 0.4rem;
        border: 1px solid rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.045);
    }

    .result-emoji {
        font-size: 4.2rem;
        line-height: 1;
    }

    .result-label {
        font-size: 1.7rem;
        font-weight: 800;
        margin-top: 0.5rem;
        letter-spacing: -0.01em;
    }

    .result-conf {
        color: #9aa0b4;
        font-size: 0.95rem;
        margin-top: 0.25rem;
    }

    /* ---------- Misc ---------- */
    hr, .stDivider {
        border-color: rgba(255,255,255,0.08) !important;
    }

    .stExpander {
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        background: rgba(255,255,255,0.03) !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="badge">TF-IDF · Logistic Regression</div>
        <h1>🎭 Emotion Detector</h1>
        <p>Type a sentence and see which emotion it carries</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This application classifies free text into one of "
        "**6 emotions**: Joy, Sadness, Anger, Fear, Love, and Surprise."
    )

    st.divider()
    st.subheader("Pipeline")
    st.markdown(
        "1. Lowercase\n"
        "2. Remove punctuation / digits\n"
        "3. Remove stopwords\n"
        "4. Saved TF-IDF vectorizer\n"
        "5. Logistic Regression"
    )

    st.divider()
    st.subheader("Emotions")
    cols = st.columns(2)
    for i, emotion in enumerate(EMOTION_ORDER):
        meta = EMOTION_META[emotion]
        cols[i % 2].markdown(f"{meta['emoji']} {meta['label']}")

    st.divider()
    st.caption("Model: `NLP_Emotions_model.joblib`")
    st.caption("Vectorizer: `NLP_Emotions_vectorizer.joblib`")


# ============================================================
# LOAD MODEL
# ============================================================

try:
    with st.spinner("Loading model and TF-IDF vectorizer..."):
        pipeline = load_pipeline()
    load_error = None

except Exception as error:
    pipeline = None
    load_error = str(error)


if load_error:
    st.error(
        f"""
        ### ❌ Unable to load the model

        {load_error}

        ---

        Make sure these files are in the same folder as `app.py`:

        - `NLP_Emotions_model.joblib`
        - `NLP_Emotions_vectorizer.joblib`
        """
    )
    st.stop()


# ============================================================
# EXAMPLE SENTENCES
# ============================================================

examples = [
    "I just got promoted and I can't stop smiling!",
    "I miss my family so much it hurts",
    "He slammed the door and stormed out on me",
    "I can't believe you actually showed up, this is unreal",
    "I keep checking the locks, something feels wrong tonight",
    "Every time I see you my heart feels so full",
]


# ============================================================
# SESSION STATE
# ============================================================

if "text_input" not in st.session_state:
    st.session_state.text_input = ""


# ============================================================
# TEXT INPUT
# ============================================================

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">✍️ Type a sentence</div>', unsafe_allow_html=True)

text_input = st.text_area(
    "Your text",
    key="text_input",
    height=120,
    placeholder="e.g. I can't believe we finally made it, I'm overjoyed!",
    label_visibility="collapsed",
)

st.caption("Or try an example:")

columns = st.columns(3)
for index, example in enumerate(examples):
    button_text = example[:28] + ("…" if len(example) > 28 else "")
    if columns[index % 3].button(button_text, key=f"example_{index}", use_container_width=True):
        st.session_state.text_input = example
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
predict_clicked = st.button("Predict Emotion ✨", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:

    if not text_input.strip():
        st.warning("Please enter some text first.")

    else:
        try:
            probabilities = predict_emotion(text_input, pipeline)
            top_emotion = max(probabilities, key=probabilities.get)
            meta = EMOTION_META[top_emotion]

            st.markdown(
                f"""
                <div class="result-card" style="box-shadow: 0 0 40px {meta['color']}22;">
                    <div class="result-emoji">{meta['emoji']}</div>
                    <div class="result-label" style="color: {meta['color']};">{meta['label']}</div>
                    <div class="result-conf">{probabilities[top_emotion] * 100:.1f}% confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            sorted_items = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            labels = [EMOTION_META[e]["label"] for e, _ in sorted_items]
            values = [p * 100 for _, p in sorted_items]
            colors = [EMOTION_META[e]["color"] for e, _ in sorted_items]

            fig = go.Figure(
                go.Bar(
                    x=values,
                    y=labels,
                    orientation="h",
                    marker=dict(
                        color=colors,
                        line=dict(width=0),
                        cornerradius=8,
                    ),
                    text=[f"{v:.1f}%" for v in values],
                    textposition="outside",
                    textfont=dict(color="#e6e6e6"),
                )
            )

            fig.update_layout(
                xaxis_title="Confidence (%)",
                yaxis=dict(autorange="reversed"),
                height=340,
                margin=dict(l=10, r=10, t=20, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e6e6e6", family="Inter"),
                xaxis=dict(range=[0, 105], gridcolor="rgba(255,255,255,0.08)"),
            )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("🔍 View processed text"):
                cleaned = clean_text(text_input, pipeline["tokenizer"], pipeline["stop_words"])
                st.code(cleaned)

        except Exception as error:
            st.error(
                f"""
                ### Prediction Error

                `{error}`

                Please make sure the model and vectorizer were
                generated from the same training run.
                """
            )