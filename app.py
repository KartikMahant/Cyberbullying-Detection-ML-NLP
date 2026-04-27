import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import string
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cyberbullying Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29, #302b63, #24243e);
}
[data-testid="stSidebar"] * { color: #e0e0ff !important; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1; margin-bottom: 0.2rem;
}
.hero-sub { color: #888; font-size: 1.05rem; font-weight: 300; margin-bottom: 1.5rem; }

.metric-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 130px;
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #2a2a5a; border-radius: 14px;
    padding: 18px 20px; text-align: center;
    box-shadow: 0 4px 24px rgba(102,126,234,0.15);
}
.metric-card .val { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; color: #a78bfa; }
.metric-card .lbl { font-size: 0.78rem; color: #888; margin-top: 2px; letter-spacing: 0.05em; text-transform: uppercase; }

.section-header {
    font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: #c4b5fd;
    border-left: 4px solid #7c3aed; padding-left: 12px; margin: 28px 0 16px 0;
}

.pred-box {
    border-radius: 16px; padding: 28px 32px; text-align: center;
    margin: 16px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.pred-label { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; }
.pred-sub { font-size: 0.95rem; opacity: 0.85; margin-top: 6px; }

.overview-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #2a2a5a; border-radius: 16px;
    padding: 22px 24px; margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(102,126,234,0.1);
}
.overview-card h4 { font-family: 'Syne', sans-serif; color: #a78bfa; font-size: 1.05rem; margin: 0 0 8px 0; }
.overview-card p { color: #bbb; font-size: 0.92rem; margin: 0; line-height: 1.6; }

.info-pill {
    display: inline-block; background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.3); border-radius: 20px;
    padding: 4px 14px; font-size: 0.82rem; color: #c4b5fd; margin: 3px;
}
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #4a3f8a, transparent);
    margin: 24px 0;
}

.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #2a2a5a; }
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif; font-size: 0.9rem; font-weight: 600;
    color: #888; background: transparent; border-radius: 8px 8px 0 0; padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    color: #a78bfa !important; border-bottom: 2px solid #7c3aed !important;
    background: rgba(124,58,237,0.08) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Load Artifacts
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("best_model.pkl",       "rb") as f: model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f: tfidf = pickle.load(f)
    with open("label_encoder.pkl",    "rb") as f: le    = pickle.load(f)
    with open("model_metadata.json",  "r")  as f: meta  = json.load(f)
    return model, tfidf, le, meta

@st.cache_data
def load_dataset():
    df = pd.read_csv("new_tweets.csv")
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    df["tweet_length"] = df["tweet_text"].astype(str).apply(len)
    df["word_count"]   = df["tweet_text"].astype(str).apply(lambda x: len(x.split()))
    return df

model, tfidf, le, meta = load_artifacts()
df = load_dataset()

# ─────────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
stop_words -= {"no", "not", "nor", "never", "neither", "nobody", "nothing"}

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
LABEL_EMOJI = {
    "gender":              "⚧️ Gender-based",
    "religion":            "🕌 Religion-based",
    "age":                 "👴 Age-based",
    "ethnicity":           "🌍 Ethnicity-based",
    "not_cyberbullying":   "✅ Not Cyberbullying",
    "other_cyberbullying": "⚠️ Other Cyberbullying",
}
LABEL_COLOR = {
    "gender":              "#e74c3c",
    "religion":            "#8e44ad",
    "age":                 "#e67e22",
    "ethnicity":           "#c0392b",
    "not_cyberbullying":   "#27ae60",
    "other_cyberbullying": "#f39c12",
}
LABEL_DESC = {
    "gender":              "Targets someone based on their gender or sexual identity.",
    "religion":            "Targets someone based on their religious beliefs.",
    "age":                 "Age-based discrimination or mockery.",
    "ethnicity":           "Racial or ethnic-based hostility.",
    "not_cyberbullying":   "This tweet does not contain cyberbullying content.",
    "other_cyberbullying": "Cyberbullying not fitting a specific category.",
}
PALETTE  = ["#667eea", "#f093fb", "#f5576c", "#4facfe", "#43e97b", "#fa709a"]
DARK_BG  = "#0d0d1a"
CARD_BG  = "#12122a"
GRID_CLR = "#1e1e3a"
TEXT_CLR = "#c4b5fd"

class_counts = df["cyberbullying_type"].value_counts()

def dark_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=10)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TEXT_CLR)
    for spine in ax.spines.values(): spine.set_edgecolor(GRID_CLR)
    ax.grid(True, color=GRID_CLR, linestyle="--", linewidth=0.5, alpha=0.7)
    return fig, ax

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ CyberGuard NLP")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Overview", "📊 EDA", "🔍 Prediction"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.82rem;color:#9090cc;line-height:2'>
    <b>Model:</b> {meta['best_model_name']}<br>
    <b>Accuracy:</b> {meta['accuracy']*100:.2f}%<br>
    <b>F1 Macro:</b> {meta['f1_macro']:.4f}<br>
    <b>F1 Weighted:</b> {meta['f1_weighted']:.4f}<br>
    <b>Classes:</b> 6<br>
    <b>Dataset:</b> ~38K tweets
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem;color:#555;text-align:center'>Built with sklearn · TF-IDF · SMOTE</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="hero-title">Cyberbullying Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">An end-to-end NLP system to classify tweets into cyberbullying categories using Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card"><div class="val">{len(df):,}</div><div class="lbl">Total Tweets</div></div>
        <div class="metric-card"><div class="val">6</div><div class="lbl">Classes</div></div>
        <div class="metric-card"><div class="val">{meta['accuracy']*100:.1f}%</div><div class="lbl">Accuracy</div></div>
        <div class="metric-card"><div class="val">{meta['f1_macro']:.3f}</div><div class="lbl">F1 Macro</div></div>
        <div class="metric-card"><div class="val">30K</div><div class="lbl">TF-IDF Features</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📋 Project Overview</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="overview-card"><h4>🎯 Objective</h4>
        <p>Automatically detect and classify cyberbullying in tweets into 6 categories:
        gender, religion, age, ethnicity, other cyberbullying, and not cyberbullying.
        Helps platforms moderate harmful content at scale.</p></div>""", unsafe_allow_html=True)

        st.markdown("""<div class="overview-card"><h4>🧹 Preprocessing Pipeline</h4>
        <p>1. Lowercase conversion<br>
        2. Remove URLs, @mentions, #hashtag symbols<br>
        3. Remove HTML entities & digits<br>
        4. Strip punctuation<br>
        5. Stopword removal (preserving negations)<br>
        6. WordNet Lemmatization</p></div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="overview-card"><h4>⚙️ Feature Engineering</h4>
        <p><b>TF-IDF Vectorizer:</b><br>
        • 30,000 max features<br>
        • Unigram + Bigram (1,2)<br>
        • Sublinear TF scaling<br>
        • min_df=2, max_df=0.95</p></div>""", unsafe_allow_html=True)

        st.markdown("""<div class="overview-card"><h4>⚖️ Imbalance Handling</h4>
        <p>• <b>SMOTE</b> — Synthetic oversampling on TF-IDF matrix<br>
        • <b>class_weight='balanced'</b> — penalty proportional to class frequency<br>
        • <b>Stratified K-Fold</b> (5 folds) for cross-validation</p></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">🤖 Models Trained & Compared</div>', unsafe_allow_html=True)
    model_info = [
        ("Logistic Regression",    "Best for TF-IDF. Linear, fast, interpretable.",  "⭐ Best"),
        ("Random Forest",          "Ensemble trees. balanced_subsample weighting.",   ""),
        ("Multinomial NB",         "Classic probabilistic text model.",               ""),
        ("Hist Gradient Boosting", "sklearn's fast boosting, class_weight support.",  ""),
    ]
    cols = st.columns(4)
    for col, (name, desc, badge) in zip(cols, model_info):
        with col:
            b = f'<span style="background:#7c3aed;border-radius:8px;padding:2px 8px;font-size:0.7rem;color:white;">{badge}</span>' if badge else ""
            st.markdown(f'<div class="overview-card" style="min-height:140px;"><h4>{name} {b}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">🏆 Best Model Configuration</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        params_html = "<br>".join([f"<b>{k}</b>: {v}" for k, v in meta['best_params'].items()])
        st.markdown(f"""<div class="overview-card"><h4>{meta['best_model_name']}</h4>
        <p>{params_html}<br><br>
        <b>Accuracy:</b> {meta['accuracy']*100:.2f}%<br>
        <b>F1 Macro:</b> {meta['f1_macro']:.4f}<br>
        <b>F1 Weighted:</b> {meta['f1_weighted']:.4f}</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="overview-card"><h4>📦 Saved Artifacts</h4>
        <p>📄 <b>best_model.pkl</b> — Trained classifier<br>
        📄 <b>tfidf_vectorizer.pkl</b> — Fitted TF-IDF transformer<br>
        📄 <b>label_encoder.pkl</b> — Label encoder (int ↔ class name)<br>
        📄 <b>model_metadata.json</b> — Accuracy, F1, params<br>
        📄 <b>new_tweets.csv</b> — Original dataset<br><br>
        Loaded at startup via <code>@st.cache_resource</code> for zero-latency inference.</p></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  PAGE 2 — EDA
# ═══════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown('<div class="hero-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Visual analysis of the tweet dataset — distributions, text patterns, and vocabulary</div>', unsafe_allow_html=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["📊 Class Distribution", "📏 Text Length", "☁️ Word Clouds", "🔤 Top Words", "📈 Stats"])

    # ── Tab 1 ──
    with t1:
        st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = dark_fig(7, 5)
            bars = ax.barh(class_counts.index, class_counts.values,
                           color=PALETTE[:len(class_counts)], edgecolor="none", height=0.6)
            ax.set_xlabel("Count"); ax.set_title("Tweet Count per Class", fontsize=13, fontweight="bold", color=TEXT_CLR)
            ax.invert_yaxis()
            for bar, val in zip(bars, class_counts.values):
                ax.text(val + 50, bar.get_y() + bar.get_height()/2, f"{val:,}", va="center", color=TEXT_CLR, fontsize=10)
            ax.set_xlim(0, class_counts.max() * 1.18)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            fig, ax = dark_fig(6, 5)
            fig.patch.set_facecolor(DARK_BG)
            wedges, texts, autotexts = ax.pie(
                class_counts.values, labels=class_counts.index, autopct="%1.1f%%",
                startangle=140, colors=PALETTE[:len(class_counts)],
                wedgeprops=dict(width=0.65, edgecolor=DARK_BG, linewidth=2)
            )
            for t in texts: t.set_color(TEXT_CLR); t.set_fontsize(9)
            for a in autotexts: a.set_color("white"); a.set_fontsize(9); a.set_fontweight("bold")
            ax.set_title("Class Proportion (%)", fontsize=13, fontweight="bold", color=TEXT_CLR)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        ratio = class_counts.max() / class_counts.min()
        st.info(f"**Imbalance Ratio (max/min):** {ratio:.2f}x — Handled with SMOTE + class_weight='balanced'.")
        count_df = pd.DataFrame({
            "Class": class_counts.index,
            "Count": class_counts.values,
            "Percentage (%)": (class_counts.values / len(df) * 100).round(2)
        })
        st.dataframe(count_df.style.background_gradient(subset=["Count"], cmap="Purples"), use_container_width=True)

    # ── Tab 2 ──
    with t2:
        st.markdown('<div class="section-header">Tweet Length Analysis</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = dark_fig(7, 4)
            ax.hist(df["tweet_length"], bins=60, color="#667eea", edgecolor=DARK_BG, linewidth=0.4, alpha=0.85)
            ax.axvline(df["tweet_length"].mean(), color="#f093fb", linestyle="--", linewidth=2, label=f"Mean: {df['tweet_length'].mean():.0f}")
            ax.axvline(df["tweet_length"].median(), color="#43e97b", linestyle="--", linewidth=2, label=f"Median: {df['tweet_length'].median():.0f}")
            ax.set_title("Character Length Distribution", fontsize=12, fontweight="bold", color=TEXT_CLR)
            ax.set_xlabel("Character Count"); ax.set_ylabel("Frequency")
            ax.legend(facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            fig, ax = dark_fig(7, 4)
            ax.hist(df["word_count"], bins=50, color="#f5576c", edgecolor=DARK_BG, linewidth=0.4, alpha=0.85)
            ax.axvline(df["word_count"].mean(), color="#f093fb", linestyle="--", linewidth=2, label=f"Mean: {df['word_count'].mean():.0f}")
            ax.axvline(df["word_count"].median(), color="#43e97b", linestyle="--", linewidth=2, label=f"Median: {df['word_count'].median():.0f}")
            ax.set_title("Word Count Distribution", fontsize=12, fontweight="bold", color=TEXT_CLR)
            ax.set_xlabel("Word Count"); ax.set_ylabel("Frequency")
            ax.legend(facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<div class="section-header">Length by Class (Box Plots)</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = dark_fig(8, 5)
            data_by_class = [df[df["cyberbullying_type"] == cls]["tweet_length"].values for cls in class_counts.index]
            bp = ax.boxplot(data_by_class, patch_artist=True, medianprops=dict(color="white", linewidth=2))
            for patch, color in zip(bp["boxes"], PALETTE): patch.set_facecolor(color); patch.set_alpha(0.7)
            for elem in ["whiskers", "caps", "fliers"]:
                for item in bp[elem]: item.set_color(TEXT_CLR)
            ax.set_xticklabels(class_counts.index, rotation=25, ha="right", fontsize=9)
            ax.set_title("Character Length by Class", fontsize=12, fontweight="bold", color=TEXT_CLR)
            ax.set_ylabel("Characters")
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            fig, ax = dark_fig(8, 5)
            data_by_class_w = [df[df["cyberbullying_type"] == cls]["word_count"].values for cls in class_counts.index]
            bp = ax.boxplot(data_by_class_w, patch_artist=True, medianprops=dict(color="white", linewidth=2))
            for patch, color in zip(bp["boxes"], PALETTE): patch.set_facecolor(color); patch.set_alpha(0.7)
            for elem in ["whiskers", "caps", "fliers"]:
                for item in bp[elem]: item.set_color(TEXT_CLR)
            ax.set_xticklabels(class_counts.index, rotation=25, ha="right", fontsize=9)
            ax.set_title("Word Count by Class", fontsize=12, fontweight="bold", color=TEXT_CLR)
            ax.set_ylabel("Words")
            fig.tight_layout(); st.pyplot(fig); plt.close()

    # ── Tab 3: Word Clouds ──
    with t3:
        st.markdown('<div class="section-header">Word Clouds by Class</div>', unsafe_allow_html=True)

        @st.cache_data
        def get_cleaned_text(cls):
            texts = df[df["cyberbullying_type"] == cls]["tweet_text"].astype(str).tolist()
            return preprocess_text(" ".join(texts))

        selected_class = st.selectbox("Select Class for Large Word Cloud", list(class_counts.index))
        wc_text = get_cleaned_text(selected_class)
        wc = WordCloud(
            width=1000, height=430, background_color="#0d0d1a",
            colormap="cool", max_words=120, collocations=False,
            prefer_horizontal=0.85
        ).generate(wc_text)
        fig, ax = plt.subplots(figsize=(13, 5))
        fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        ax.set_title(f"Word Cloud — {LABEL_EMOJI.get(selected_class, selected_class)}",
                     color=TEXT_CLR, fontsize=14, fontweight="bold", pad=12)
        fig.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<div class="section-header">All Classes — Mini Word Clouds</div>', unsafe_allow_html=True)
        cmaps = ["cool", "plasma", "hot", "spring", "YlOrRd", "winter"]
        cols = st.columns(3)
        for i, cls in enumerate(class_counts.index):
            txt = get_cleaned_text(cls)
            if txt.strip():
                wc_mini = WordCloud(
                    width=400, height=220, background_color="#12122a",
                    colormap=cmaps[i % len(cmaps)], max_words=60, collocations=False
                ).generate(txt)
                fig, ax = plt.subplots(figsize=(5, 2.8))
                fig.patch.set_facecolor("#12122a"); ax.set_facecolor("#12122a")
                ax.imshow(wc_mini, interpolation="bilinear"); ax.axis("off")
                ax.set_title(LABEL_EMOJI.get(cls, cls), color=TEXT_CLR, fontsize=9, fontweight="bold")
                fig.tight_layout(pad=0.3)
                with cols[i % 3]: st.pyplot(fig)
                plt.close()

    # ── Tab 4: Top Words ──
    with t4:
        st.markdown('<div class="section-header">Top Words per Class</div>', unsafe_allow_html=True)
        n_words = st.slider("Number of top words", 5, 25, 15)

        @st.cache_data
        def get_top_words(cls, n):
            txt = get_cleaned_text(cls)
            return Counter(txt.split()).most_common(n)

        classes = list(class_counts.index)
        for i in range(0, len(classes), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(classes):
                    cls = classes[i + j]
                    top = get_top_words(cls, n_words)
                    if top:
                        words, counts = zip(*top)
                        with col:
                            fig, ax = dark_fig(6, 4)
                            color = LABEL_COLOR.get(cls, "#7c3aed")
                            ax.barh(list(reversed(words)), list(reversed(counts)),
                                    color=color, edgecolor="none", height=0.65)
                            ax.set_title(LABEL_EMOJI.get(cls, cls), fontsize=11, fontweight="bold", color=TEXT_CLR)
                            ax.set_xlabel("Frequency")
                            fig.tight_layout(); st.pyplot(fig); plt.close()

    # ── Tab 5: Stats ──
    with t5:
        st.markdown('<div class="section-header">Dataset Statistics</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Character Length by Class**")
            st.dataframe(
                df.groupby("cyberbullying_type")["tweet_length"].describe().round(1)
                  .style.background_gradient(cmap="Purples"), use_container_width=True)
        with col2:
            st.markdown("**Word Count by Class**")
            st.dataframe(
                df.groupby("cyberbullying_type")["word_count"].describe().round(1)
                  .style.background_gradient(cmap="Blues"), use_container_width=True)

        st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(5, 3.5)
        corr = df[["tweet_length", "word_count"]].corr()
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", ax=ax,
                    linewidths=1, linecolor=DARK_BG, annot_kws={"color": "white", "fontsize": 13})
        ax.set_title("Feature Correlation", color=TEXT_CLR, fontsize=12, fontweight="bold")
        ax.tick_params(colors=TEXT_CLR)
        fig.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<div class="section-header">Data Quality</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", f"{len(df):,}")
        col2.metric("Null Values", int(df.isnull().sum().sum()))
        col3.metric("Duplicate Rows", int(df.duplicated().sum()))

        st.markdown('<div class="section-header">Sample Tweets</div>', unsafe_allow_html=True)
        sample_cls = st.selectbox("Filter by class", ["All"] + list(class_counts.index))
        sample = df if sample_cls == "All" else df[df["cyberbullying_type"] == sample_cls]
        st.dataframe(sample[["tweet_text", "cyberbullying_type"]].sample(min(8, len(sample)), random_state=1), use_container_width=True)


# ═══════════════════════════════════════════════
#  PAGE 3 — PREDICTION
# ═══════════════════════════════════════════════
elif page == "🔍 Prediction":
    st.markdown('<div class="hero-title">Live Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Enter any tweet and instantly classify its cyberbullying type</div>', unsafe_allow_html=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px;">
        <span class="info-pill">🤖 {meta['best_model_name']}</span>
        <span class="info-pill">🎯 Accuracy: {meta['accuracy']*100:.2f}%</span>
        <span class="info-pill">📊 F1 Macro: {meta['f1_macro']:.4f}</span>
        <span class="info-pill">⚙️ {' | '.join([f"{k}={v}" for k, v in meta['best_params'].items()])}</span>
    </div>""", unsafe_allow_html=True)

    # ✅ INPUT ONLY (no columns now)
    st.markdown('<div class="section-header">📝 Enter Tweet</div>', unsafe_allow_html=True)

    tweet_input = st.text_area(
        "tweet",
        height=150,
        placeholder="Type or paste tweet...",
        label_visibility="collapsed"
    )

    analyze_btn = st.button("🔍 Analyze Tweet", use_container_width=True, type="primary")

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ✅ PREDICTION LOGIC
    if analyze_btn and tweet_input.strip():
        cleaned  = preprocess_text(tweet_input)
        vec      = tfidf.transform([cleaned])
        pred_idx = model.predict(vec)[0]
        label    = le.inverse_transform([pred_idx])[0]
        color    = LABEL_COLOR.get(label, "#7c3aed")
        proba    = model.predict_proba(vec)[0] if hasattr(model, "predict_proba") else None

        col_res, col_prob = st.columns([2, 3])

        with col_res:
            st.markdown(f"""
            <div class="pred-box" style="background:linear-gradient(135deg,{color}cc,{color}44);border:2px solid {color};">
                <div style="font-size:2.5rem;">{LABEL_EMOJI.get(label,'').split()[0]}</div>
                <div class="pred-label">{label.replace('_',' ').upper()}</div>
                <div class="pred-sub">{LABEL_DESC.get(label,'')}</div>
            </div>""", unsafe_allow_html=True)

            with st.expander("🔎 Cleaned text fed to model"):
                st.code(cleaned if cleaned else "(empty after preprocessing)", language="text")

        with col_prob:
            if proba is not None:
                st.markdown('<div class="section-header">Class Probabilities</div>', unsafe_allow_html=True)

                sorted_proba = sorted(zip(le.classes_, proba), key=lambda x: x[1], reverse=True)

                fig, ax = plt.subplots(figsize=(7, 3.8))
                fig.patch.set_facecolor(DARK_BG)
                ax.set_facecolor(CARD_BG)

                classes_sorted = [LABEL_EMOJI.get(c, c) for c, _ in sorted_proba]
                probs_sorted   = [p for _, p in sorted_proba]
                bar_colors     = [LABEL_COLOR.get(c, "#7c3aed") for c, _ in sorted_proba]

                bars = ax.barh(classes_sorted, probs_sorted, color=bar_colors, edgecolor="none", height=0.55)

                ax.set_xlim(0, 1.18)
                ax.set_xlabel("Probability", color=TEXT_CLR, fontsize=9)
                ax.tick_params(colors=TEXT_CLR, labelsize=9)

                for spine in ax.spines.values():
                    spine.set_edgecolor(GRID_CLR)

                ax.grid(True, axis="x", color=GRID_CLR, linestyle="--", linewidth=0.5)
                ax.invert_yaxis()

                for bar, prob in zip(bars, probs_sorted):
                    ax.text(prob + 0.01, bar.get_y() + bar.get_height()/2,
                            f"{prob*100:.1f}%", va="center", color="white", fontsize=9, fontweight="bold")

                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

    elif analyze_btn:
        st.warning("⚠️ Please enter some tweet text before clicking Analyze.")