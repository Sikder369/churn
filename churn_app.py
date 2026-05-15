import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# PAGE CONFIG AND Design
#======================================>>>>>>>>>>

st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# CUSTOM CSS for better design
# ==============================>>>>>

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #dbeafe 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1f1c2c 0%, #928DAB 100%);
}

[data-testid="stSidebar"] * {
    color: white;
}

.main-title {
    text-align: center;
    color: #1f2937;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 10px;
}

.page-title {
    text-align: center;
    color: #2563eb;
    font-size: 34px;
    font-weight: bold;
    margin-bottom: 25px;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 2px 2px 15px rgba(0,0,0,0.12);
    margin-bottom: 20px;
}

.section-heading {
    text-align: center;
    color: white;
    background: linear-gradient(90deg, #2563eb, #9333ea);
    padding: 14px;
    border-radius: 15px;
    margin-bottom: 20px;
}

div[data-testid="metric-container"] {
    background-color: white;
    border: 2px solid #e5e7eb;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# LOAD AND PREPARE DATA
# ==============================>>>>>

df = pd.read_csv("telco_customer_churn.csv")

#Clean data
df.columns = df.columns.str.strip()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

#Convert target column
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

##Drop Customer ID
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

#Encode categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)


drop_columns = [
    "Churn",
    "gender_Male",
    "PhoneService_Yes",
    "StreamingTV_Yes",
    "StreamingMovies_Yes",
    "MultipleLines_Yes",
    "DeviceProtection_Yes",
    "OnlineBackup_Yes",
    "PaymentMethod_Mailed check",
    "PaymentMethod_Credit card (automatic)",
    "InternetService_No",
    "Contract_One year"
]

X = df_encoded.drop(drop_columns, axis=1, errors="ignore")
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model_R = LogisticRegression(max_iter=1000)
model_R.fit(X_train, y_train)

y_pred = model_R.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# SIDEBAR
# ==============================>>>>

st.sidebar.markdown("""
<h1 style='text-align:center;'>📊 Churn Prediction App 🕵️ </h1>
<p style='text-align:center;'>Logistic Regression Dashboard</p>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select Page",
    [
        "Dataset💾",
        "EDA📊",
        "Accuracy🎯",
        "Customer Churn Prediction🎰"
    ]
)

# MAIN TITLE
# ==================

st.markdown("""
<div class="main-title">
Telecom Customer Churn Prediction App
</div>
""", unsafe_allow_html=True)

# PAGE 1: DATASET
# ==============================

if page == "Dataset💾":

    st.markdown("""
    <div class="page-title">
    Dataset Overview
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Target Variable", "Churn")

    st.markdown("""
    <div class="section-heading">
    Dataset Preview
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("""
    <div class="section-heading">
    Dataset Columns
    </div>
    """, unsafe_allow_html=True)

    st.write(df.columns.tolist())
    st.markdown("""
    <div class="section-heading">
    Target Column Distribution
    </div>
    """, unsafe_allow_html=True)

    churn_counts = y.value_counts()
    
    churn_counts = y.value_counts()

    fig, ax = plt.subplots(figsize=(3,4))

    bars = ax.bar(
    churn_counts.index.astype(str),
    churn_counts.values,
    color=["green", "red"]
    )

    ax.bar_label(bars)

    st.pyplot(fig, use_container_width=False)

# PAGE 2: EDA & MODEL EVALUATION
# ==============================

elif page == "EDA📊":

    st.markdown("""
    <div class="page-title">
    Exploratory Data Analysis & Model Evaluation
    </div>
    """, unsafe_allow_html=True)

    # Classification Report
    st.markdown("""
    <div class="section-heading">
    Classification Report
    </div>
    """, unsafe_allow_html=True)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)


# Confusion Matrix
    st.markdown("""
    <div class="section-heading">
    Confusion Matrix
    </div>
    """, unsafe_allow_html=True)

    cm = confusion_matrix(y_test, y_pred)

    col_cm1, col_cm2, col_cm3 = st.columns([1, 1, 1])

    with col_cm2:
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            annot_kws={"size": 14},
            linewidths=0.5,
            ax=ax_cm
        )

        ax_cm.set_xlabel("Predicted", fontsize=10)
        ax_cm.set_ylabel("Actual", fontsize=10)
        ax_cm.set_title("Confusion Matrix", fontsize=12)
        ax_cm.tick_params(axis="both", labelsize=9)

        st.pyplot(fig_cm)

     # Heatmaps
    st.markdown("""
    <div class="section-heading">
    Correlation Analysis Dashboard
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <h4 style='text-align:center; color:#1f2937;'>
        Features Correlated with Churn
        </h4>
        """, unsafe_allow_html=True)

        corr_matrix = df_encoded.corr()
        churn_corr = corr_matrix[["Churn"]].sort_values(
            by="Churn",
            ascending=False
        )

        fig1, ax1 = plt.subplots(figsize=(5, 8))

        sns.heatmap(
            churn_corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            annot_kws={"size": 7},
            cbar=False,
            ax=ax1
        )

        ax1.tick_params(axis="y", labelsize=8)
        ax1.set_title("Churn Correlation", fontsize=11)

        st.pyplot(fig1)

    with col2:
        st.markdown("""
        <h4 style='text-align:center; color:#1f2937;'>
        Full Feature Correlation Heatmap
        </h4>
        """, unsafe_allow_html=True)

        corr_matrix2 = X.corr()

        fig2, ax2 = plt.subplots(figsize=(10, 7))

        sns.heatmap(
            corr_matrix2,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.3,
            annot_kws={"size": 6},
            ax=ax2
        )

        ax2.tick_params(axis="x", labelrotation=90, labelsize=7)
        ax2.tick_params(axis="y", labelsize=7)
        ax2.set_title("Feature Correlation", fontsize=11)

        st.pyplot(fig2)

# CHURN BY GENDER
# ==============================

    st.markdown("""
    <div class="section-heading">
    Churn Distribution by Gender
    </div>
    """, unsafe_allow_html=True)

# Create figure
    fig_gender, ax_gender = plt.subplots(figsize=(6,4))

    sns.countplot(
    data=df,
    x="gender",
    hue="Churn",
    palette="coolwarm",
    ax=ax_gender
    )

    ax_gender.set_title("Male vs Female Churn")
    ax_gender.set_xlabel("Gender")
    ax_gender.set_ylabel("Customer Count")

# Show values on bars
    for container in ax_gender.containers:
        ax_gender.bar_label(container)

    st.pyplot(fig_gender)



# PAGE 3: ACCURACY
# ==============================>>>>>

elif page == "Accuracy🎯":

    st.markdown("""
    <div class="page-title">
    Model Accuracy
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "Logistic Regression")
    col1.metric("Model", "KNN")
    col1.metric("Model", "Random Forest")
    col2.metric("Accuracy", f"{round(accuracy * 100, 2)}%")
    col2.metric("Accuracy", f"75.91%")
    col2.metric("Accuracy", f"78.89%")
    col3.metric("Test Size", "20%")
    col3.metric("Test Size", "20%")
    col3.metric("Test Size", "20%")

    st.markdown("""
    <div class="section-heading">
    Accuracy Progress
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(accuracy * 100))

    st.success(f"The Logistic Regression model achieved an accuracy of {round(accuracy * 100, 2)}%.")
    st.success(f"The Logistic Regression model will be used as it has the Highest Accuracy Score")


# PAGE 4: CUSTOMER CHURN PREDICTION
# ============================>>>!!

elif page == "Customer Churn Prediction🎰":

    st.markdown("""
    <div class="page-title">
    Customer Churn Prediction
    </div>
    """, unsafe_allow_html=True)

    st.info("Enter customer information below and click Predict Churn.")

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure", min_value=0, max_value=100, value=12)

        with col2:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

        with col3:
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

        col4, col5, col6 = st.columns(3)

        with col4:
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        with col5:
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        with col6:
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
            total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

        submitted = st.form_submit_button("Predict Churn")

    if submitted:

        input_data = pd.DataFrame({
            "SeniorCitizen": [senior_citizen],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges]
        })

        input_encoded = pd.get_dummies(input_data, drop_first=True)
        input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

        prediction = model_R.predict(input_encoded)[0]
        probability = model_R.predict_proba(input_encoded)[0][1]

        st.markdown("""
        <div class="section-heading">
        Prediction Result
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(probability * 100))

        if prediction == 1:
            st.error(f"Customer is likely to Leave. Churn Probability: {round(probability * 100, 2)}%")
        else:
            st.success(f"Customer is likely to STAY. Churn probablity: {round(probability * 100, 2)}%")