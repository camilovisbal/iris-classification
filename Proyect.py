import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

st.set_page_config(page_title="Iris Classification", layout="wide")

st.markdown("""
<style>
    .header-block {
        background-color: #1a3a5c;
        padding: 18px 28px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: white;
    }
    .header-block h2 { margin: 0 0 4px 0; font-size: 1.4rem; }
    .header-block p { margin: 2px 0; font-size: 0.88rem; opacity: 0.85; }
    div[data-testid="metric-container"] {
        background: #f7f9fc;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-block">
    <h2>Iris Species Classification</h2>
    <p><strong>University:</strong> Universidad de la Costa &nbsp;|&nbsp; <strong>Course:</strong> Data Mining</p>
    <p><strong>Professor:</strong> Jose Escorcia-Gutierrez, Ph.D.</p>
    <p><strong>Students:</strong> Camilo Andres Visbal Beltran &nbsp;|&nbsp; Julian Paba de la Hoz &nbsp;|&nbsp; Adrian Llamas Quintero &nbsp;|&nbsp; Roberto De los Reyes Castro</p>
</div>
""", unsafe_allow_html=True)

DATA_URL = "https://raw.githubusercontent.com/camilovisbal/iris-classification/main/Iris.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df.drop(columns=["Id"], inplace=True)
    return df

df = load_data()

color_map = {
    "Iris-setosa": "#2e86ab",
    "Iris-versicolor": "#e84855",
    "Iris-virginica": "#3bb273"
}

with st.sidebar:
    st.header("Filters")
    selected_species = st.multiselect(
        "Species",
        options=sorted(df["Species"].unique().tolist()),
        default=sorted(df["Species"].unique().tolist())
    )
    sepal_range = st.slider(
        "Sepal Length (cm)",
        float(df["SepalLengthCm"].min()),
        float(df["SepalLengthCm"].max()),
        (float(df["SepalLengthCm"].min()), float(df["SepalLengthCm"].max()))
    )
    petal_range = st.slider(
        "Petal Length (cm)",
        float(df["PetalLengthCm"].min()),
        float(df["PetalLengthCm"].max()),
        (float(df["PetalLengthCm"].min()), float(df["PetalLengthCm"].max()))
    )

filtered = df[
    df["Species"].isin(selected_species) &
    df["SepalLengthCm"].between(*sepal_range) &
    df["PetalLengthCm"].between(*petal_range)
]

# train model on the full dataset
le = LabelEncoder()
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = le.fit_transform(df["Species"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted")
rec  = recall_score(y_test, y_pred, average="weighted")
f1   = f1_score(y_test, y_pred, average="weighted")

st.subheader("Model Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy",  f"{acc:.4f}")
c2.metric("Precision", f"{prec:.4f}")
c3.metric("Recall",    f"{rec:.4f}")
c4.metric("F1 Score",  f"{f1:.4f}")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Scatter Matrix", "3D View & Prediction", "Confusion Matrix"])

with tab1:
    feat = st.selectbox(
        "Feature",
        options=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    )
    fig_hist = px.histogram(
        filtered, x=feat, color="Species", barmode="overlay",
        color_discrete_map=color_map, nbins=25,
        title=f"Distribution of {feat} by Species",
        labels={feat: feat.replace("Cm", " (cm)")}
    )
    fig_hist.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_hist, use_container_width=True)

    fig_box = px.box(
        filtered, x="Species", y=feat, color="Species",
        color_discrete_map=color_map,
        title=f"Boxplot of {feat} by Species"
    )
    fig_box.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    fig_matrix = px.scatter_matrix(
        filtered, dimensions=features, color="Species",
        color_discrete_map=color_map,
        title="Scatter Matrix — All Features"
    )
    fig_matrix.update_traces(diagonal_visible=False, marker=dict(size=4, opacity=0.7))
    fig_matrix.update_layout(height=600)
    st.plotly_chart(fig_matrix, use_container_width=True)

with tab3:
    st.markdown("#### Predict a new sample")
    p1, p2 = st.columns(2)
    with p1:
        s_len = st.number_input("Sepal Length (cm)", min_value=1.0, max_value=10.0, value=5.1, step=0.1)
        s_wid = st.number_input("Sepal Width (cm)",  min_value=1.0, max_value=10.0, value=3.5, step=0.1)
    with p2:
        p_len = st.number_input("Petal Length (cm)", min_value=0.1, max_value=10.0, value=1.4, step=0.1)
        p_wid = st.number_input("Petal Width (cm)",  min_value=0.1, max_value=10.0, value=0.2, step=0.1)

    if st.button("Predict"):
        sample = np.array([[s_len, s_wid, p_len, p_wid]])
        pred_label = le.inverse_transform(clf.predict(sample))[0]
        proba = clf.predict_proba(sample)[0]
        st.success(f"Predicted species: **{pred_label}**")
        prob_df = pd.DataFrame({"Species": le.classes_, "Probability": proba})
        st.dataframe(prob_df.set_index("Species").style.format("{:.4f}"), use_container_width=True)

        new_point = pd.DataFrame({
            "SepalLengthCm": [s_len], "SepalWidthCm": [s_wid],
            "PetalLengthCm": [p_len], "PetalWidthCm": [p_wid],
            "Species": ["New Sample"]
        })
        plot_df = pd.concat([filtered, new_point], ignore_index=True)
        cmap_ext = {**color_map, "New Sample": "#f4a261"}

        fig_3d = px.scatter_3d(
            plot_df, x="SepalLengthCm", y="PetalLengthCm", z="PetalWidthCm",
            color="Species", color_discrete_map=cmap_ext, symbol="Species",
            title="3D Scatter — New Sample Position",
            labels={"SepalLengthCm": "Sepal Length", "PetalLengthCm": "Petal Length", "PetalWidthCm": "Petal Width"}
        )
        fig_3d.update_traces(marker=dict(size=4))
        for trace in fig_3d.data:
            if trace.name == "New Sample":
                trace.marker.size = 10
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        fig_3d = px.scatter_3d(
            filtered, x="SepalLengthCm", y="PetalLengthCm", z="PetalWidthCm",
            color="Species", color_discrete_map=color_map,
            title="3D Scatter Plot",
            labels={"SepalLengthCm": "Sepal Length", "PetalLengthCm": "Petal Length", "PetalWidthCm": "Petal Width"}
        )
        fig_3d.update_traces(marker=dict(size=4))
        st.plotly_chart(fig_3d, use_container_width=True)

with tab4:
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm, x=le.classes_, y=le.classes_,
        colorscale="Blues", showscale=True,
        text=cm, texttemplate="%{text}"
    ))
    fig_cm.update_layout(
        title="Confusion Matrix (test set)",
        xaxis_title="Predicted", yaxis_title="Actual",
        plot_bgcolor="white", paper_bgcolor="white", height=420
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    fi_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": clf.feature_importances_
    }).sort_values("Importance", ascending=True)
    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        title="Feature Importance (Random Forest)",
        color="Importance", color_continuous_scale="Blues"
    )
    fig_fi.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
    st.plotly_chart(fig_fi, use_container_width=True)

st.markdown("---")
with st.expander("View filtered dataset"):
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True)
    st.caption(f"{len(filtered)} records shown out of {len(df)} total.")
