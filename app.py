# ============================================================
# CELL 1 — Install libraries
# ============================================================
# Colab mein pehle yeh cell run karo
!pip install -q streamlit sentence-transformers pyngrok matplotlib seaborn scikit-learn


# ============================================================
# CELL 2 — app.py file banao
# ============================================================
app_code = '''
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="NLP Similarity Lab", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    .main-title { font-size:2.2rem; font-weight:700; color:#6366f1; }
    .ct-card { background:#f8fafc; border-left:4px solid #6366f1;
               border-radius:0 8px 8px 0; padding:0.9rem 1.1rem; margin:0.4rem 0; font-size:0.9rem; }
    .ct-label { font-weight:700; color:#4f46e5; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🔬 NLP Similarity Lab</p>', unsafe_allow_html=True)
st.caption("Model: all-MiniLM-L6-v2 · SentenceTransformers · No preprocessing · No training")
st.markdown("---")

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

with st.spinner("Loading pretrained model..."):
    model = load_model()

st.sidebar.header("⚙️ Input Panel")
default_texts = (
    "The cat sat on the mat.\n"
    "A dog rested on the carpet.\n"
    "Machine learning is a subset of AI.\n"
    "Deep learning uses neural networks.\n"
    "The sun rises in the east.\n"
    "I love playing football."
)
raw_input = st.sidebar.text_area("Enter sentences (one per line):", value=default_texts, height=220)
sentences = [s.strip() for s in raw_input.strip().split("\\n") if s.strip()]
query = st.sidebar.text_input("🔍 Query sentence:", value="Artificial intelligence is transforming the world.")
top_n = st.sidebar.slider("Top N results:", 2, min(8, len(sentences)), min(5, len(sentences)))

if len(sentences) < 2:
    st.warning("Please enter at least 2 sentences.")
    st.stop()

with st.spinner("Encoding..."):
    embeddings  = model.encode(sentences, convert_to_tensor=True)
    query_emb   = model.encode(query, convert_to_tensor=True)

cos_scores  = util.cos_sim(query_emb, embeddings)[0].cpu().numpy()
pair_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
np_emb      = embeddings.cpu().numpy()

ranked_idx    = np.argsort(cos_scores)[::-1][:top_n]
ranked_scores = cos_scores[ranked_idx]
ranked_sents  = [sentences[i] for i in ranked_idx]
short_labels  = [f"S{i+1}" for i in range(len(sentences))]

st.subheader("📊 Similarity Results")
st.markdown(f"**Query:** _{query}_")
st.markdown(f"**Top match:** _{ranked_sents[0]}_ — score: **{ranked_scores[0]:.4f}**")

# ── Graph 1: Bar Chart ────────────────────────────────────────
st.subheader("📈 Graph 1 · Top Similar Sentences (Bar Chart)")
fig1, ax1 = plt.subplots(figsize=(9, 3.8))
labels = [
    (f"S{ranked_idx[i]+1}: {ranked_sents[i][:45]}…"
     if len(ranked_sents[i]) > 45
     else f"S{ranked_idx[i]+1}: {ranked_sents[i]}")
    for i in range(len(ranked_idx))
]
bars = ax1.barh(labels, ranked_scores, color="#6366f1", edgecolor="white", height=0.55)
for bar, score in zip(bars, ranked_scores):
    ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             f"{score:.4f}", va="center", fontsize=9)
ax1.set_xlim(0, 1.12)
ax1.set_xlabel("Cosine Similarity Score")
ax1.set_title(f"Top {top_n} matches for query", fontweight="bold")
ax1.invert_yaxis()
ax1.spines[["top","right"]].set_visible(False)
fig1.tight_layout()
st.pyplot(fig1)

# ── Graph 2: Heatmap ──────────────────────────────────────────
st.subheader("🟥 Graph 2 · Pairwise Similarity Heatmap")
fig2, ax2 = plt.subplots(figsize=(max(6, len(sentences)), max(5, len(sentences)-1)))
sns.heatmap(pair_matrix, annot=True, fmt=".2f",
            xticklabels=short_labels, yticklabels=short_labels,
            cmap="RdYlGn", vmin=0, vmax=1,
            linewidths=0.5, ax=ax2, annot_kws={"size":9})
ax2.set_title("Pairwise Cosine Similarity Between All Sentences", fontweight="bold")
fig2.tight_layout()
st.pyplot(fig2)
with st.expander("Label legend"):
    for i, s in enumerate(sentences):
        st.markdown(f"**S{i+1}** → {s}")

# ── Graph 3: PCA Plot ─────────────────────────────────────────
st.subheader("🧭 Graph 3 · 2D Embedding Space (PCA)")
pca    = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(np_emb)
fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.scatter(coords[:,0], coords[:,1],
            c=range(len(sentences)), cmap="tab10",
            s=120, edgecolors="white", linewidth=1.2, zorder=3)
for i,(x,y) in enumerate(coords):
    ax3.annotate(short_labels[i], (x,y),
                 textcoords="offset points", xytext=(8,4), fontsize=9)
ax3.set_title("PCA – embeddings projected to 2D", fontweight="bold")
ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax3.spines[["top","right"]].set_visible(False)
ax3.grid(alpha=0.2)
fig3.tight_layout()
st.pyplot(fig3)

# ── Critical Thinking ─────────────────────────────────────────
st.subheader("🧠 Paul\\\'s Critical Thinking Standards")
top_score  = ranked_scores[0]
top_sent   = ranked_sents[0]
second     = ranked_scores[1] if len(ranked_scores) > 1 else None

ct_items = [
    ("🔍 Clarity",
     f"The user provided {len(sentences)} sentences. The model encoded each into a 384-dim vector. "
     "Cosine similarity (0=unrelated, 1=identical) ranks semantic closeness to the query."),
    ("✅ Accuracy",
     "Model used: <b>all-MiniLM-L6-v2</b> (SentenceTransformers / Hugging Face). "
     "No training, fine-tuning, or preprocessing was applied."),
    ("🎯 Precision",
     f"Top score: <b>{top_score:.4f}</b> for: <i>\"{top_sent}\"</i>. "
     + (f"2nd score: <b>{second:.4f}</b> (diff: {abs(top_score-second):.4f})." if second else "")),
    ("📌 Relevance",
     "Bar chart ranks query matches; heatmap shows cross-sentence clusters; "
     "PCA confirms semantically similar sentences occupy nearby vector space."),
    ("💡 Logic",
     "The top sentence scores highest because it shares the most semantic concepts with the query "
     "in the embedding space learned during pretraining on 1B+ sentence pairs."),
    ("⭐ Significance",
     f"Most significant result: score <b>{top_score:.4f}</b>. "
     "Scores >0.75 = strong overlap; <0.30 = unrelated. "
     "PCA visually confirms semantic clusters without any labeling."),
    ("⚖️ Fairness",
     "<b>Limitation:</b> all-MiniLM-L6-v2 is English-centric. "
     "Performance drops for non-English text, domain-specific jargon, or very short inputs. "
     "Scores are relative rankings, not absolute semantic distances."),
]
for label, text in ct_items:
    st.markdown(
        f\'<div><span class="ct-label">{label}</span><br>{text}</div>\',
        unsafe_allow_html=True
    )

st.markdown("---")
st.caption("NLP Similarity Lab · Shifa Tameer-e-Millat University · all-MiniLM-L6-v2 · No preprocessing · No training")
'''

with open("app.py", "w") as f:
    f.write(app_code)

print("✅ app.py created!")


# ============================================================
# CELL 3 — ngrok auth token set karo
# (https://dashboard.ngrok.com/ pe free account banao, token copy karo)
# ============================================================
from pyngrok import ngrok

ngrok.set_auth_token("3Fo3BusXS1wkGX35El5MEsftCm8_HwAtsb6FGVeA96EWbNsk")   # <-- yahan apna token paste karo


# ============================================================
# CELL 4 — Streamlit run karo + public URL lo
# ============================================================
import subprocess, time

proc = subprocess.Popen(
    ["streamlit", "run", "app.py",
     "--server.port", "8501",
     "--server.headless", "true"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)

time.sleep(5)   # app start hone do

public_url = ngrok.connect(8501)
print("=" * 50)
print(f"✅ App chal rahi hai!")
print(f"🌐 Public URL:  {public_url}")
print("=" * 50)
print("Jab kaam khatam ho to CELL band karo ya proc.terminate() chalao.")
