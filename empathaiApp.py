"""
Project: EmpathAI - GenAI Powered Mental Health Chatbot (Lightweight Version)
Description:
    - Detects emotion from user text (Transformers)
    - Detects basic risk level (self-harm / crisis keywords)
    - Retrieves relevant self-help snippets using semantic similarity (SentenceTransformer)
    - Generates empathetic responses using smart templates (no GPT-2 → faster)
    - Applies a safety layer for high-risk messages

Run:
    streamlit run empathai_app.py
"""

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from streamlit.components.v1 import html
import numpy as np
import textwrap

# -------------------------------------------------------------------
# 1. CONFIG & CONSTANTS
# -------------------------------------------------------------------

EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Simple rule-based crisis keywords for risk classification
CRISIS_KEYWORDS = [
    "suicide",
    "kill myself",
    "end my life",
    "self harm",
    "cut myself",
    "die",
    "worthless",
    "no reason to live",
]

# Small in-memory "vector database" of validated self-help snippets
SELF_HELP_SNIPPETS = [
    {
        "id": 1,
        "category": "anxiety",
        "text": (
            "Try the 4-7-8 breathing exercise: inhale for 4 seconds, "
            "hold for 7 seconds, and exhale slowly for 8 seconds. "
            "Repeat this 4–6 times to calm your nervous system."
        ),
    },
    {
        "id": 2,
        "category": "grounding",
        "text": (
            "Use the 5-4-3-2-1 grounding technique: name 5 things you can see, "
            "4 things you can feel, 3 things you can hear, 2 things you can smell, "
            "and 1 thing you can taste."
        ),
    },
    {
        "id": 3,
        "category": "journaling",
        "text": (
            "Take a notebook and write down what you are feeling in this moment "
            "without judging yourself. Label the emotion and what triggered it."
        ),
    },
    {
        "id": 4,
        "category": "self-compassion",
        "text": (
            "Talk to yourself like you would talk to a close friend who is struggling. "
            "Be gentle, avoid harsh self-criticism, and remind yourself that it is okay "
            "to have difficult days."
        ),
    },
    {
        "id": 5,
        "category": "stress",
        "text": (
            "Take a short break from screens, stretch your body for a few minutes, "
            "drink some water, and step outside if possible. Small physical shifts "
            "can reduce mental stress."
        ),
    },
]

# -------------------------------------------------------------------
# 2. LOAD MODELS (CACHED IN STREAMLIT)
# -------------------------------------------------------------------

@st.cache_resource(show_spinner=True)
def load_emotion_classifier():
    """
    Loads a transformer-based emotion classifier.
    """
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME)
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        top_k=None,
    )
    return pipe


@st.cache_resource(show_spinner=True)
def load_embedding_model():
    """
    Loads sentence-transformer for semantic embeddings.
    """
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model


@st.cache_resource(show_spinner=True)
def build_vector_store():
    """
    Builds simple in-memory vector store from SELF_HELP_SNIPPETS.
    Returns:
        embeddings_matrix: np.array of shape (N, D)
        ids: list of snippet ids
    """
    emb_model = load_embedding_model()
    texts = [item["text"] for item in SELF_HELP_SNIPPETS]
    embeddings = emb_model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    )
    ids = [item["id"] for item in SELF_HELP_SNIPPETS]
    return embeddings, ids

# -------------------------------------------------------------------
# 3. CORE LOGIC FUNCTIONS
# -------------------------------------------------------------------

def classify_emotion(text: str):
    """
    Uses transformer model to classify emotion and get scores.
    Returns:
        dominant_emotion (str), score (float), full_scores (list of dict)
    """
    emotion_pipe = load_emotion_classifier()
    outputs = emotion_pipe(text)[0]  # list of dicts: [{'label': 'joy', 'score': 0.8}, ...]
    outputs = sorted(outputs, key=lambda x: x["score"], reverse=True)
    dominant = outputs[0]
    return dominant["label"], dominant["score"], outputs


def detect_risk(text: str):
    """
    Simple rule-based risk detection.
    Returns:
        risk_level: 'low', 'medium', or 'high'
        matched_keywords: list of keywords that triggered risk
    """
    text_lower = text.lower()
    matched = [kw for kw in CRISIS_KEYWORDS if kw in text_lower]

    if matched:
        return "high", matched
    return "low", []


def retrieve_self_help_snippets(user_text: str, top_k: int = 3):
    """
    RAG-style retrieval: find top_k snippets semantically similar to user_text.
    """
    emb_model = load_embedding_model()
    vec_store, ids = build_vector_store()

    query_emb = emb_model.encode(
        user_text, convert_to_numpy=True, normalize_embeddings=True
    )
    sims = np.dot(vec_store, query_emb)
    top_indices = sims.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        snippet = next(item for item in SELF_HELP_SNIPPETS if item["id"] == ids[idx])
        results.append(
            {
                "id": snippet["id"],
                "category": snippet["category"],
                "text": snippet["text"],
                "similarity": float(sims[idx]),
            }
        )
    return results


def generate_empathetic_response(
    user_text: str,
    emotion: str,
    risk_level: str,
    retrieved_snippets: list,
):
    """
    Generates empathetic response (template-based, fast).
    If risk_level == 'high' -> crisis-safety message.
    Otherwise: emotion-aware supportive message using retrieved tips.
    """
    if risk_level == "high":
        crisis_msg = textwrap.dedent(
            """
            I’m really glad you reached out and I’m so sorry that you’re feeling this way.  
            Your safety is the most important thing right now.

            I’m not a medical professional, but please consider these steps immediately:
            • If you are in immediate danger, contact your local emergency number right away.
            • Reach out to a trusted person (family, friend, teacher, or mentor) and tell them how you feel.
            • You can also look for local mental health helplines, suicide prevention lines, or hospital hotlines.

            You do not have to go through this alone. Reaching out for help is a strong and brave step. 💛
            """
        ).strip()
        return crisis_msg

    tips_text = "\n".join(
        [f"- {item['text']}" for item in retrieved_snippets]
    ) or "- Take a small, gentle step to care for yourself today."

    response = textwrap.dedent(
        f"""
        Thank you for sharing how you feel. It sounds like you may be experiencing some **{emotion}**, 
        and that can be really heavy to carry.

        Your feelings are valid, and it makes sense that you are reacting this way to what you're going through.  
        You deserve patience and kindness, especially from yourself.

        Here are a few gentle ideas you might try:
        {tips_text}

        This chatbot is just a supportive tool and not a replacement for a therapist or doctor,  
        but you absolutely deserve real support. If your feelings become overwhelming, please consider 
        talking to a counselor, mental health professional, or someone you deeply trust. 💛
        """
    ).strip()

    return response

# -------------------------------------------------------------------
# 4. STREAMLIT UI
# -------------------------------------------------------------------

def main():
    st.set_page_config(page_title="EmpathAI - Mental Health Chatbot", page_icon="💛")

    # ---------- Custom CSS for background & cards ----------
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #020617 100%);
            color: #e5e7eb;
        }
        .main > div {
            padding-top: 10px;
        }
        .chat-card {
            background: rgba(15, 23, 42, 0.9);
            border-radius: 16px;
            padding: 16px 20px;
            border: 1px solid rgba(148, 163, 184, 0.4);
        }
        .glass-card {
            background: rgba(15, 23, 42, 0.7);
            border-radius: 16px;
            padding: 16px 20px;
            border: 1px solid rgba(148, 163, 184, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Popup on first load ----------
    if "popup_shown" not in st.session_state:
        st.session_state["popup_shown"] = False

    if not st.session_state["popup_shown"]:
        html(
            """
            <script>
            alert("This website is made by Ribhu Bhushan Tiwari");
            </script>
            """,
            height=0,
        )
        st.session_state["popup_shown"] = True

    # ---------- Main content ----------
    st.markdown(
        "<h1 style='text-align: center; color: #e5e7eb;'>💛 EmpathAI: Mental Health Chatbot</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #cbd5f5;'>A GenAI-inspired chatbot that detects emotions, "
        "retrieves gentle self-help tips, and responds with empathy.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Settings")
        show_debug = st.checkbox("Show debug details (for viva / report)", value=True)
        st.markdown(
            "<small style='color:#cbd5f5;'>Note: This is an academic prototype and not a medical tool.</small>",
            unsafe_allow_html=True,
        )

    st.subheader("🗨️ Share what you’re feeling")

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    user_text = st.text_area(
        "Type your message here:",
        placeholder="Example: I'm feeling very anxious and overwhelmed about my studies and life...",
        height=150,
        label_visibility="collapsed",
    )
    send_button = st.button("Send 💬", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if send_button:
        if not user_text.strip():
            st.warning("Please type something before sending.")
            return

        with st.spinner("Thinking with empathy... 🧠"):
            # 1. Emotion classification
            emotion, emo_score, emo_full = classify_emotion(user_text)

            # 2. Risk detection
            risk_level, matched_keywords = detect_risk(user_text)

            # 3. Retrieval of self-help snippets
            retrieved = retrieve_self_help_snippets(user_text, top_k=3)

            # 4. Template-based empathetic response (fast)
            response = generate_empathetic_response(
                user_text=user_text,
                emotion=emotion,
                risk_level=risk_level,
                retrieved_snippets=retrieved,
            )

        # ---------------- DISPLAY RESULTS ----------------

        st.markdown("### 💬 EmpathAI's Response")
        st.markdown("<div class='chat-card'>", unsafe_allow_html=True)
        st.markdown(response)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎭 Detected Emotion")
            st.write(f"**Dominant emotion:** `{emotion}` (confidence: `{emo_score:.2f}`)")
            if show_debug:
                st.write("All emotion scores:")
                for item in emo_full:
                    st.write(f"- {item['label']}: {item['score']:.2f}")

        with col2:
            st.markdown("#### 🚨 Risk Level")
            color_text = {
                "low": "✅ Low",
                "medium": "🟠 Medium",
                "high": "⛔ High",
            }[risk_level]
            st.write(f"**Risk level:** {color_text}")
            if matched_keywords:
                st.write("Matched crisis keywords:")
                for kw in matched_keywords:
                    st.write(f"- `{kw}`")

        st.markdown("### 🌱 Suggested Self-Help Snippets (RAG)")
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        for item in retrieved:
            st.markdown(
                f"- **[{item['category'].title()}]** {item['text']}  "
                f"(similarity: `{item['similarity']:.2f}`)"
            )
        st.markdown("</div>", unsafe_allow_html=True)

        if show_debug:
            st.markdown("---")
            st.markdown("### 🧪 Tech Notes (for Viva / Report)")
            st.write(
                "- Emotion classifier: transformer-based model "
                f"(`{EMOTION_MODEL_NAME}`)\n"
                "- Embeddings: SentenceTransformer "
                f"(`{EMBEDDING_MODEL_NAME}`)\n"
                "- Retrieval: cosine similarity over snippet embeddings "
                "(simple in-memory vector store)\n"
                "- Response: template-based, emotion-aware + RAG tips\n"
                "- Safety layer: crisis keyword detection → overrides normal response"
            )

if __name__ == "__main__":
    main()
