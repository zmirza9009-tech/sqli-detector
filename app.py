import streamlit as st
import joblib, os

st.title("SQL Injection Detector")

@st.cache_resource
def load_models():
    models = {}
    for name, fname in [("Naive Bayes","model_naivebayes.pkl"),("Logistic Regression","model_logistic.pkl"),("Random Forest","model_randomforest.pkl"),("SVM","model_svm.pkl")]:
        if os.path.exists(fname):
            d = joblib.load(fname)
            models[name] = d
    return models

models = load_models()
if not models:
    st.error("No model files found!")
    st.stop()

query = st.text_area("Type a SQL query to test:", height=80)
mode = st.radio("Mode:", ["Test one model", "Compare all models"], horizontal=True)

if mode == "Test one model":
    chosen = st.selectbox("Pick a model:", list(models.keys()))

if st.button("Analyze", type="primary"):
    if not query.strip():
        st.warning("Please type a query first.")
    else:
        def predict(mdata, q):
            proba = mdata['model'].predict_proba([q])[0]
            idx = proba.argmax()
            return mdata['label_encoder'].classes_[idx], proba[idx], dict(zip(mdata['label_encoder'].classes_, proba))

        if mode == "Test one model":
            label, conf, scores = predict(models[chosen], query)
            if label == "Benign":
                st.success(f"SAFE — {label} ({conf*100:.1f}% confidence)")
            else:
                st.error(f"ATTACK DETECTED — {label} ({conf*100:.1f}% confidence)")
            st.write("---")
            for cls, sc in sorted(scores.items(), key=lambda x: -x[1]):
                st.write(f"{'🟢' if cls=='Benign' else '🔴'} {cls}: {sc*100:.1f}%")
                st.progress(float(sc))
        else:
            st.write("### All models results:")
            cols = st.columns(len(models))
            for col, (mname, mdata) in zip(cols, models.items()):
                label, conf, _ = predict(mdata, query)
                with col:
                    st.write(f"**{mname}**")
                    if label == "Benign":
                        st.success(f"SAFE\n{conf*100:.1f}%")
                    else:
                        st.error(f"ATTACK\n{label}\n{conf*100:.1f}%")