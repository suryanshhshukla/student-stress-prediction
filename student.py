"""Student Stress Predictor: a Streamlit app built on the notebook's cleaning and models.

Run with:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance

DATA_FILE = Path(__file__).parent / "student-lifestyle-and-stress-dataset.csv"
SEED = 42

NUM_COLS = ["Sleep_Hours", "Study_Hours", "Social_Media_Hours",
            "Attendance", "Exam_Pressure", "Family_Support"]
NUMERIC = NUM_COLS + ["Month"]
CATEGORICAL = ["Student_Type"]
FEATURES = CATEGORICAL + NUMERIC

NICE = {
    "Sleep_Hours": "Sleep hours",
    "Study_Hours": "Study hours",
    "Social_Media_Hours": "Social media hours",
    "Attendance": "Attendance",
    "Exam_Pressure": "Exam pressure",
    "Family_Support": "Family support",
    "Month": "Month",
    "Student_Type": "Student type",
}
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
TYPE_LABELS = {"school": "School student", "college": "College student",
               "working_student": "Working student"}

RED, BLUE = "#E45756", "#4C78A8"


# ----------------------------------------------------------------------------
# Data and models
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load the CSV and apply the same cleaning rules as the notebook."""
    df = pd.read_csv(DATA_FILE)
    df.loc[(df["Study_Hours"] < 0) | (df["Study_Hours"] > 16), "Study_Hours"] = np.nan
    df.loc[(df["Attendance"] <= 0) | (df["Attendance"] > 100), "Attendance"] = np.nan
    return df.drop_duplicates().reset_index(drop=True)


def make_preprocessor(scale):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])


@st.cache_resource(show_spinner="Training the models (only the first time)...")
def train_models():
    df = load_data()
    X, y = df[FEATURES], df["Stress_Level"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    models = {
        "Logistic Regression": Pipeline([
            ("prep", make_preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "Random Forest": Pipeline([
            ("prep", make_preprocessor(scale=False)),
            ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=10,
                                           class_weight="balanced", n_jobs=-1,
                                           random_state=SEED)),
        ]),
        "Gradient Boosting": Pipeline([
            ("prep", make_preprocessor(scale=False)),
            ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                   max_depth=4, class_weight="balanced",
                                                   random_state=SEED)),
        ]),
    }

    rows, probas = [], {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        probas[name] = proba
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred),
            "Recall": recall_score(y_te, pred),
            "F1": f1_score(y_te, pred),
            "ROC-AUC": roc_auc_score(y_te, proba),
        })
    metrics = pd.DataFrame(rows).set_index("Model").round(3)

    gb_pred = (probas["Gradient Boosting"] >= 0.5).astype(int)
    cm = confusion_matrix(y_te, gb_pred)

    perm = permutation_importance(models["Gradient Boosting"], X_te, y_te,
                                  scoring="roc_auc", n_repeats=5,
                                  random_state=SEED, n_jobs=-1)
    importance = pd.Series(perm.importances_mean, index=X_te.columns).sort_values()

    return {"models": models, "metrics": metrics, "cm": cm, "importance": importance}


def predict_student(models, student):
    """Return (gradient boosting probability, logistic probability, factor contributions)."""
    row = pd.DataFrame([student])[FEATURES]
    p_gb = float(models["Gradient Boosting"].predict_proba(row)[0, 1])
    lr = models["Logistic Regression"]
    p_lr = float(lr.predict_proba(row)[0, 1])

    # Logistic regression: coefficient x standardized value = push relative to the average student
    x = lr.named_steps["prep"].transform(row)
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    coefs = lr.named_steps["clf"].coef_[0]
    n = len(NUMERIC)
    contrib = pd.Series(coefs[:n] * x[0, :n], index=[NICE[c] for c in NUMERIC])
    return p_gb, p_lr, contrib.sort_values()


def risk_level(p):
    if p < 0.35:
        return "Low", "#2E7D32", "Low stress risk"
    if p < 0.65:
        return "Moderate", "#EF8F00", "Moderate stress risk"
    return "High", "#C62828", "High stress risk"


def build_tips(s):
    tips = []
    if s["Sleep_Hours"] < 6:
        tips.append("Sleep is on the low side. Aim for 7-8 hours; in this data, students who sleep more are less often stressed.")
    if s["Exam_Pressure"] >= 7:
        tips.append("Exam pressure is the single biggest factor in this data. Breaking revision into small daily goals and talking to a teacher about the schedule can help.")
    if s["Study_Hours"] >= 8:
        tips.append("Very long study days go with higher stress. Short breaks and a fixed stopping time help.")
    if s["Social_Media_Hours"] >= 5:
        tips.append("Social media time is high. Even a small reduction can free up time for sleep and rest.")
    if s["Family_Support"] <= 3:
        tips.append("Support matters. Talking to a friend, mentor or counsellor can make up for low family support.")
    if s["Attendance"] < 65:
        tips.append("Attendance is low, which can add catch-up pressure later. Try to stay on top of classes.")
    if not tips:
        tips.append("Nothing stands out as a strong stress factor here. Keep up the healthy routine.")
    return tips


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
CSS = """
<style>
.hero {background: linear-gradient(120deg, #4C78A8 0%, #7B5EA7 100%);
       padding: 1.6rem 2rem; border-radius: 16px; color: white; margin-bottom: 1.2rem;}
.hero h1 {color: white; margin: 0; font-size: 2rem;}
.hero p {margin: .4rem 0 0 0; opacity: .92; font-size: 1.02rem;}
.result {padding: 1.2rem 1.4rem; border-radius: 14px; color: white; text-align: center;}
.result .big {font-size: 3rem; font-weight: 700; line-height: 1.1;}
.result .label {font-size: 1.1rem; opacity: .95;}
.tip {background: #F4F6FA; border-left: 4px solid #4C78A8; padding: .6rem .9rem;
      border-radius: 6px; margin-bottom: .5rem; color: #1F2937;}
</style>
"""


def sidebar_inputs(df):
    st.sidebar.header("Student profile")
    stype = st.sidebar.selectbox("Student type", list(TYPE_LABELS),
                                 format_func=TYPE_LABELS.get, index=1)
    sleep = st.sidebar.slider("Sleep hours per night", 2.0, 10.0, 6.5, 0.1)
    study = st.sidebar.slider("Study hours per day", 0.0, 16.0, 4.5, 0.1)
    social = st.sidebar.slider("Social media hours per day", 0.0, 10.0, 3.5, 0.1)
    attendance = st.sidebar.slider("Attendance (%)", 40, 100, 82)
    exam = st.sidebar.slider("Exam pressure (1 = low, 10 = high)", 1, 10, 5)
    family = st.sidebar.slider("Family support (1 = low, 10 = high)", 1, 10, 6)
    month = st.sidebar.selectbox("Month", range(1, 13), format_func=lambda m: MONTHS[m - 1], index=5)
    st.sidebar.caption("Move the sliders and the prediction updates instantly.")
    return {"Student_Type": stype, "Sleep_Hours": sleep, "Study_Hours": study,
            "Social_Media_Hours": social, "Attendance": float(attendance),
            "Exam_Pressure": float(exam), "Family_Support": float(family),
            "Month": float(month)}


def tab_predict(models, student):
    p_gb, p_lr, contrib = predict_student(models, student)
    level, color, text = risk_level(p_gb)

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown(
            f"<div class='result' style='background:{color}'>"
            f"<div class='label'>{text}</div>"
            f"<div class='big'>{p_gb:.0%}</div>"
            f"<div class='label'>estimated chance of being stressed</div></div>",
            unsafe_allow_html=True)
        st.progress(min(max(p_gb, 0.0), 1.0))
        c1, c2 = st.columns(2)
        c1.metric("Gradient Boosting", f"{p_gb:.0%}")
        c2.metric("Logistic Regression", f"{p_lr:.0%}")
        st.caption("Two different models, shown as a cross-check. If they roughly agree, the estimate is more reliable.")

    with right:
        st.subheader("What pushes stress up or down?")
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        colors = [RED if v > 0 else BLUE for v in contrib.values]
        ax.barh(contrib.index, contrib.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Effect compared with an average student")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Red bars raise the stress estimate, blue bars lower it (from the logistic regression model).")

    st.subheader("Suggestions")
    for tip in build_tips(student):
        st.markdown(f"<div class='tip'>{tip}</div>", unsafe_allow_html=True)


def tab_explore(df):
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{len(df):,}")
    c2.metric("Stressed", f"{df['Stress_Level'].mean():.0%}")
    c3.metric("Features", len(FEATURES))
    c4.metric("Missing values", f"{int(df.isna().sum().sum()):,}")

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Stress rate by student type (%)**")
        rate = (df.groupby("Student_Type")["Stress_Level"].mean() * 100).round(1)
        rate.index = [TYPE_LABELS[i] for i in rate.index]
        st.bar_chart(rate, color=RED)
    with b:
        st.markdown("**Stress rate by month (%)**")
        by_month = (df.groupby("Month")["Stress_Level"].mean() * 100).round(1)
        by_month.index = [MONTHS[int(m) - 1][:3] for m in by_month.index]
        st.line_chart(by_month, color=BLUE)

    st.markdown("**Stress rate by exam pressure (%)**")
    ep = (df.groupby("Exam_Pressure")["Stress_Level"].mean() * 100).round(1)
    st.bar_chart(ep, color=RED)

    st.markdown("**Compare stressed and not stressed students**")
    feature = st.selectbox("Choose a feature", NUM_COLS, format_func=NICE.get)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for label, name, col in [(0, "Not stressed", BLUE), (1, "Stressed", RED)]:
        vals = df.loc[df["Stress_Level"] == label, feature].dropna()
        ax.hist(vals, bins=25, alpha=0.55, color=col, label=name, density=True)
    ax.set_xlabel(NICE[feature])
    ax.set_ylabel("Share of students")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Show a sample of the data"):
        st.dataframe(df.head(100), use_container_width=True)


def tab_model(results):
    st.subheader("How good are the models?")
    st.caption("Scores on 20% of the students that the models never saw during training.")
    metrics = results["metrics"].sort_values("ROC-AUC", ascending=False)
    st.dataframe(metrics, use_container_width=True)

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Confusion matrix (Gradient Boosting)**")
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        ConfusionMatrixDisplay(results["cm"], display_labels=["Not stressed", "Stressed"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.grid(False)
        st.pyplot(fig)
        plt.close(fig)
    with b:
        st.markdown("**Feature importance**")
        imp = results["importance"]
        fig, ax = plt.subplots(figsize=(4.8, 3.8))
        ax.barh([NICE[i] for i in imp.index], imp.values, color=BLUE)
        ax.set_xlabel("Drop in ROC-AUC when shuffled")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        "**Reading the results:** exam pressure is by far the strongest factor, followed by study hours. "
        "Recall tells how many truly stressed students the model catches; precision tells how many of its "
        "'stressed' predictions are right. Because only about 30% of students are stressed, accuracy alone is misleading.")


def main():
    st.set_page_config(page_title="Student Stress Predictor", page_icon="🧠", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        "<div class='hero'><h1>🧠 Student Stress Predictor</h1>"
        "<p>Estimate a student's stress level from lifestyle and academic habits using machine learning.</p></div>",
        unsafe_allow_html=True)

    if not DATA_FILE.exists():
        st.error(f"Dataset not found. Put `{DATA_FILE.name}` in the same folder as app.py.")
        st.stop()

    df = load_data()
    results = train_models()
    student = sidebar_inputs(df)

    t1, t2, t3 = st.tabs(["🔮 Predict", "📊 Explore the data", "🤖 Model performance"])
    with t1:
        tab_predict(results["models"], student)
    with t2:
        tab_explore(df)
    with t3:
        tab_model(results)

    st.divider()
    st.caption("For learning purposes only. This is a statistical model trained on a dataset, "
               "not a medical or psychological diagnosis. If you or someone you know is struggling, "
               "please talk to a counsellor, doctor or someone you trust.")


if __name__ == "__main__":
    main()
