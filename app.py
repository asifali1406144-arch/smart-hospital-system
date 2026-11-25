# ============================================================
# 🏥 SMART HOSPITAL SYSTEM
# Data Mining + Streamlit
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix

from mlxtend.frequent_patterns import apriori, association_rules


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Hospital System",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f8fc;
}

.block-container {
    padding-top: 1.5rem;
}

h1 {
    color: #12355b;
}

h2, h3 {
    color: #1f4e79;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.title("🏥 Smart Hospital Management & Data Mining System")

st.markdown(
    "### Intelligent Patient Analysis, Disease Prediction & Data Mining Dashboard"
)

st.divider()


# ============================================================
# GENERATE HOSPITAL DATA
# ============================================================

@st.cache_data
def generate_data(n=500):

    np.random.seed(42)

    ages = np.random.randint(18, 81, n)

    genders = np.random.choice(
        ["Male", "Female"],
        n
    )

    blood_pressure = np.random.randint(
        90, 181, n
    )

    glucose = np.random.randint(
        70, 221, n
    )

    cholesterol = np.random.randint(
        120, 301, n
    )

    symptoms = np.random.choice(
        [
            "Fever",
            "Cough",
            "Headache",
            "Chest Pain",
            "Fatigue",
            "Shortness of Breath",
            "Dizziness"
        ],
        n
    )

    diseases = []

    for i in range(n):

        score = 0

        if ages[i] > 55:
            score += 1

        if blood_pressure[i] > 140:
            score += 1

        if glucose[i] > 140:
            score += 1

        if cholesterol[i] > 220:
            score += 1

        if symptoms[i] in [
            "Chest Pain",
            "Shortness of Breath"
        ]:
            score += 1

        if score >= 3:
            disease = "High Risk"
        elif score == 2:
            disease = "Moderate Risk"
        else:
            disease = "Low Risk"

        diseases.append(disease)

    df = pd.DataFrame({
        "Patient_ID": [
            f"P{1001+i}"
            for i in range(n)
        ],
        "Age": ages,
        "Gender": genders,
        "Blood_Pressure": blood_pressure,
        "Glucose": glucose,
        "Cholesterol": cholesterol,
        "Main_Symptom": symptoms,
        "Risk_Level": diseases
    })

    return df


df = generate_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏥 Hospital Menu")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Patient Records",
        "Data Visualization",
        "Disease Prediction",
        "Patient Clustering",
        "Association Rules"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Hospital Overview")

    total_patients = len(df)

    high_risk = len(
        df[df["Risk_Level"] == "High Risk"]
    )

    moderate_risk = len(
        df[df["Risk_Level"] == "Moderate Risk"]
    )

    low_risk = len(
        df[df["Risk_Level"] == "Low Risk"]
    )

    average_age = round(
        df["Age"].mean(),
        1
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "👥 Total Patients",
        total_patients
    )

    col2.metric(
        "⚠️ High Risk",
        high_risk
    )

    col3.metric(
        "🟡 Moderate Risk",
        moderate_risk
    )

    col4.metric(
        "📅 Average Age",
        average_age
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        risk_counts = df["Risk_Level"].value_counts()

        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Patient Risk Distribution",
            hole=0.4
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        gender_counts = df["Gender"].value_counts()

        fig = px.bar(
            x=gender_counts.index,
            y=gender_counts.values,
            title="Patients by Gender",
            labels={
                "x": "Gender",
                "y": "Patients"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("🚨 High Risk Patients")

    high_risk_df = df[
        df["Risk_Level"] == "High Risk"
    ]

    st.dataframe(
        high_risk_df.head(20),
        use_container_width=True
    )


# ============================================================
# PATIENT RECORDS
# ============================================================

elif page == "Patient Records":

    st.header("👥 Patient Records")

    search = st.text_input(
        "🔎 Search Patient ID"
    )

    if search:

        filtered = df[
            df["Patient_ID"]
            .str.contains(
                search,
                case=False
            )
        ]

    else:

        filtered = df

    st.dataframe(
        filtered,
        use_container_width=True
    )

    st.download_button(
        "📥 Download Patient Dataset",
        df.to_csv(index=False),
        "hospital_data.csv",
        "text/csv"
    )


# ============================================================
# DATA VISUALIZATION
# ============================================================

elif page == "Data Visualization":

    st.header("📈 Data Visualization")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="Age",
            nbins=15,
            title="Age Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.histogram(
            df,
            x="Blood_Pressure",
            nbins=15,
            title="Blood Pressure Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    fig = px.scatter(
        df,
        x="Glucose",
        y="Cholesterol",
        color="Risk_Level",
        hover_data=[
            "Patient_ID",
            "Age",
            "Gender"
        ],
        title="Glucose vs Cholesterol"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    symptom_counts = (
        df["Main_Symptom"]
        .value_counts()
        .reset_index()
    )

    symptom_counts.columns = [
        "Symptom",
        "Patients"
    ]

    fig = px.bar(
        symptom_counts,
        x="Symptom",
        y="Patients",
        title="Most Common Symptoms"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# DISEASE PREDICTION
# ============================================================

elif page == "Disease Prediction":

    st.header("🤖 Disease Risk Prediction")

    X = df[
        [
            "Age",
            "Blood_Pressure",
            "Glucose",
            "Cholesterol"
        ]
    ]

    y = df["Risk_Level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    st.success(
        f"Model Accuracy: {accuracy*100:.2f}%"
    )

    st.subheader("Enter Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=40
        )

        bp = st.number_input(
            "Blood Pressure",
            min_value=80,
            max_value=220,
            value=120
        )

    with col2:

        glucose_value = st.number_input(
            "Glucose",
            min_value=50,
            max_value=300,
            value=100
        )

        cholesterol_value = st.number_input(
            "Cholesterol",
            min_value=100,
            max_value=400,
            value=180
        )

    if st.button(
        "🔮 Predict Risk"
    ):

        input_data = pd.DataFrame({
            "Age": [age],
            "Blood_Pressure": [bp],
            "Glucose": [glucose_value],
            "Cholesterol": [cholesterol_value]
        })

        prediction = model.predict(
            input_data
        )[0]

        if prediction == "High Risk":

            st.error(
                f"⚠️ Predicted Risk: {prediction}"
            )

        elif prediction == "Moderate Risk":

            st.warning(
                f"🟡 Predicted Risk: {prediction}"
            )

        else:

            st.success(
                f"🟢 Predicted Risk: {prediction}"
            )


# ============================================================
# PATIENT CLUSTERING
# ============================================================

elif page == "Patient Clustering":

    st.header("🔍 Patient Clustering")

    st.write(
        "K-Means clustering groups patients with similar health characteristics."
    )

    features = df[
        [
            "Age",
            "Blood_Pressure",
            "Glucose",
            "Cholesterol"
        ]
    ]

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        features
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    df_cluster = df.copy()

    df_cluster["Cluster"] = (
        kmeans.fit_predict(
            scaled_data
        ) + 1
    )

    st.subheader("Clustered Patients")

    st.dataframe(
        df_cluster.head(30),
        use_container_width=True
    )

    fig = px.scatter(
        df_cluster,
        x="Age",
        y="Glucose",
        color="Cluster",
        hover_data=[
            "Patient_ID",
            "Blood_Pressure",
            "Cholesterol"
        ],
        title="Patient Clusters"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    cluster_counts = (
        df_cluster["Cluster"]
        .value_counts()
        .sort_index()
    )

    fig = px.bar(
        x=cluster_counts.index.astype(str),
        y=cluster_counts.values,
        title="Number of Patients in Each Cluster",
        labels={
            "x": "Cluster",
            "y": "Patients"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# ASSOCIATION RULES
# ============================================================

elif page == "Association Rules":

    st.header("🔗 Association Rule Mining")

    st.write(
        "This module identifies relationships between patient symptoms and risk levels."
    )

    transactions = pd.DataFrame({
        "Fever": df["Main_Symptom"].eq("Fever"),
        "Cough": df["Main_Symptom"].eq("Cough"),
        "Headache": df["Main_Symptom"].eq("Headache"),
        "Chest_Pain": df["Main_Symptom"].eq("Chest Pain"),
        "Fatigue": df["Main_Symptom"].eq("Fatigue"),
        "Shortness_of_Breath": df["Main_Symptom"].eq(
            "Shortness of Breath"
        ),
        "Dizziness": df["Main_Symptom"].eq("Dizziness"),
        "High_Risk": df["Risk_Level"].eq("High Risk"),
        "Moderate_Risk": df["Risk_Level"].eq(
            "Moderate Risk"
        ),
        "Low_Risk": df["Risk_Level"].eq("Low Risk")
    })

    frequent_items = apriori(
        transactions,
        min_support=0.05,
        use_colnames=True
    )

    if len(frequent_items) > 0:

        rules = association_rules(
            frequent_items,
            metric="confidence",
            min_threshold=0.20
        )

        if len(rules) > 0:

            rules = rules.sort_values(
                "lift",
                ascending=False
            )

            rules_display = rules[
                [
                    "antecedents",
                    "consequents",
                    "support",
                    "confidence",
                    "lift"
                ]
            ].head(20)

            st.dataframe(
                rules_display,
                use_container_width=True
            )

            st.success(
                "Association rules successfully generated."
            )

        else:

            st.warning(
                "No strong association rules were found."
            )

    else:

        st.warning(
            "No frequent patterns were found."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🏥 Smart Hospital System | Data Mining Project | "
    "Classification • Clustering • Association Rules"
)
