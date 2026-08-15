# ============================================================
# CUSTOMER CHURN PREDICTION + SHAP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #f8f9ff 0%,
        #eef2ff 50%,
        #f8f9ff 100%
    );
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ================= HERO ================= */

.hero {
    background: linear-gradient(
        135deg,
        #312e81,
        #4f46e5,
        #7c3aed
    );

    padding: 38px 42px;
    border-radius: 24px;

    color: white;

    margin-bottom: 32px;

    box-shadow:
        0 15px 40px rgba(79,70,229,0.25);
}

.hero-title {
    font-size: 40px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 17px;
    opacity: 0.92;
    line-height: 1.6;
}

.hero-badges {
    margin-top: 22px;
}

.hero-badge {
    display: inline-block;

    background: rgba(255,255,255,0.15);

    border: 1px solid rgba(255,255,255,0.25);

    padding: 8px 15px;

    border-radius: 20px;

    margin-right: 8px;

    font-size: 13px;
    font-weight: 600;
}


/* ================= SECTION TITLES ================= */

.section-title {
    font-size: 27px;
    font-weight: 750;

    color: #1e1b4b;

    margin-top: 10px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #64748b;

    font-size: 14px;

    margin-bottom: 20px;
}


/* ================= CARDS ================= */

.card {
    background: white;

    padding: 25px;

    border-radius: 20px;

    border: 1px solid #e2e8f0;

    box-shadow:
        0 8px 25px rgba(15,23,42,0.06);

    margin-bottom: 20px;
}


/* ================= METRIC CARDS ================= */

.metric-card {
    background: white;

    padding: 20px;

    border-radius: 18px;

    border: 1px solid #e2e8f0;

    box-shadow:
        0 8px 25px rgba(15,23,42,0.06);

    text-align: center;

    min-height: 125px;
}

.metric-label {
    color: #64748b;

    font-size: 12px;

    font-weight: 700;

    letter-spacing: 0.7px;

    margin-bottom: 10px;
}

.metric-value {
    color: #1e1b4b;

    font-size: 25px;

    font-weight: 800;
}


/* ================= PROBABILITY ================= */

.probability-card {
    background: linear-gradient(
        135deg,
        #eef2ff,
        #ffffff
    );

    border: 1px solid #c7d2fe;

    border-radius: 20px;

    padding: 25px;

    text-align: center;

    margin-top: 20px;

    margin-bottom: 15px;
}

.probability-label {
    color: #64748b;

    font-size: 13px;

    font-weight: 700;

    letter-spacing: 1.2px;
}

.probability-number {
    color: #4338ca;

    font-size: 46px;

    font-weight: 850;
}


/* ================= INFO ================= */

.info-box {
    background: #f8fafc;

    padding: 17px 20px;

    border-radius: 14px;

    border-left: 5px solid #6366f1;

    color: #475569;

    line-height: 1.7;
}


/* ================= SHAP ================= */

.shap-card {
    background: white;

    padding: 25px;

    border-radius: 20px;

    border: 1px solid #e2e8f0;

    box-shadow:
        0 8px 25px rgba(15,23,42,0.06);

    margin-top: 25px;
}

.shap-title {
    font-size: 25px;

    font-weight: 750;

    color: #1e1b4b;

    margin-bottom: 5px;
}

.shap-subtitle {
    color: #64748b;

    font-size: 14px;

    margin-bottom: 15px;
}


/* ================= FOOTER ================= */

.footer {
    text-align: center;

    color: #64748b;

    font-size: 13px;

    padding: 30px 0 10px 0;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = os.path.join(
        os.path.dirname(__file__),
        "customer_churn_model.pkl"
    )

    if not os.path.exists(model_path):

        st.error(
            "❌ customer_churn_model.pkl was not found "
            "inside the Model_Deployment folder."
        )

        st.stop()

    return joblib.load(model_path)


model = load_model()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">

<div class="hero-title">
📊 Customer Churn Prediction
</div>

<div class="hero-subtitle">
AI-powered customer retention intelligence using
Machine Learning to predict customer churn risk
and explain the factors behind each prediction.
</div>

<div class="hero-badges">

<span class="hero-badge">
🤖 Random Forest
</span>

<span class="hero-badge">
📈 Predictive Analytics
</span>

<span class="hero-badge">
🎯 Churn Prediction
</span>

<span class="hero-badge">
🧠 SHAP Explainability
</span>

</div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# MAIN COLUMNS
# ============================================================

input_col, result_col = st.columns(
    [1.1, 0.9],
    gap="large"
)


# ============================================================
# CUSTOMER INPUT
# ============================================================

with input_col:

    st.markdown(
        """
<div class="section-title">
👤 Customer Profile
</div>

<div class="section-subtitle">
Enter the customer's key information to estimate churn risk.
</div>
""",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # MAIN FEATURES
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        tenure = st.number_input(
            "⏳ Tenure (months)",
            min_value=0,
            max_value=100,
            value=12,
            step=1
        )

        contract = st.selectbox(
            "📄 Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        internet_service = st.selectbox(
            "🌐 Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        payment_method = st.selectbox(
            "💳 Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )


    with col2:

        monthly_charges = st.number_input(
            "💰 Monthly Charges",
            min_value=0.0,
            max_value=500.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "💵 Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=1500.0,
            step=10.0
        )

        online_security = st.selectbox(
            "🔐 Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "🛠️ Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


    # --------------------------------------------------------
    # ADDITIONAL DETAILS
    # --------------------------------------------------------

    with st.expander(
        "⚙️ Additional Customer Details"
    ):

        st.caption(
            "Additional features required by the trained model."
        )

        c1, c2, c3 = st.columns(3)


        with c1:

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x:
                "Yes" if x == 1 else "No"
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )


        with c2:

            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                [
                    "Yes",
                    "No",
                    "No phone service"
                ]
            )

            online_backup = st.selectbox(
                "Online Backup",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            device_protection = st.selectbox(
                "Device Protection",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )


        with c3:

            streaming_tv = st.selectbox(
                "Streaming TV",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )


# ============================================================
# CREATE CUSTOMER DATA
# ============================================================

customer_data = pd.DataFrame(
    {
        "gender": [gender],
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
    }
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

with result_col:

    st.markdown(
        """
<div class="section-title">
📈 Churn Intelligence
</div>

<div class="section-subtitle">
Machine Learning assessment of this customer's churn risk.
</div>
""",
        unsafe_allow_html=True
    )

    predict_button = st.button(
        "🔮 Predict Customer Churn",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        probability = model.predict_proba(
            customer_data
        )[0, 1]


        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        FINAL_THRESHOLD = 0.40

        if probability >= FINAL_THRESHOLD:

            prediction = "Will Churn"

        else:

            prediction = "Will Stay"


        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        if probability >= 0.70:

            risk = "High Risk"
            risk_color = "#dc2626"
            risk_icon = "🔴"

        elif probability >= 0.40:

            risk = "Medium Risk"
            risk_color = "#d97706"
            risk_icon = "🟠"

        else:

            risk = "Low Risk"
            risk_color = "#059669"
            risk_icon = "🟢"


        probability_percentage = probability * 100


        # ====================================================
        # RESULTS
        # ====================================================

        st.markdown(
            """
<div class="result-card">

<div class="result-title">
🎯 Prediction Result
</div>

</div>
""",
            unsafe_allow_html=True
        )


        r1, r2, r3 = st.columns(3)


        with r1:

            st.markdown(
                f"""
<div class="metric-card">

<div class="metric-label">
CHURN PROBABILITY
</div>

<div class="metric-value">
{probability_percentage:.1f}%
</div>

</div>
""",
                unsafe_allow_html=True
            )


        with r2:

            icon = (
                "⚠️"
                if prediction == "Will Churn"
                else "✅"
            )

            color = (
                "#dc2626"
                if prediction == "Will Churn"
                else "#059669"
            )

            st.markdown(
                f"""
<div class="metric-card">

<div class="metric-label">
PREDICTION
</div>

<div class="metric-value"
style="color:{color};">

{icon} {prediction}

</div>

</div>
""",
                unsafe_allow_html=True
            )


        with r3:

            st.markdown(
                f"""
<div class="metric-card">

<div class="metric-label">
RISK LEVEL
</div>

<div class="metric-value"
style="color:{risk_color};">

{risk_icon} {risk}

</div>

</div>
""",
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        st.markdown(
            f"""
<div class="probability-card">

<div class="probability-label">
CUSTOMER CHURN PROBABILITY
</div>

<div class="probability-number">
{probability_percentage:.1f}%
</div>

</div>
""",
            unsafe_allow_html=True
        )

        st.progress(
            float(probability)
        )


        # ----------------------------------------------------
        # RISK MESSAGE
        # ----------------------------------------------------

        if risk == "High Risk":

            st.error(
                "🔴 High churn risk — proactive retention action "
                "should be considered."
            )

        elif risk == "Medium Risk":

            st.warning(
                "🟠 Medium churn risk — targeted engagement "
                "may be useful."
            )

        else:

            st.success(
                "🟢 Low churn risk — this customer currently "
                "appears relatively stable."
            )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.markdown(
            """
<div class="shap-card">

<div class="shap-title">
🧠 Why did the model make this prediction?
</div>

<div class="shap-subtitle">
SHAP shows how individual customer features influenced
the model's churn prediction.
</div>

</div>
""",
            unsafe_allow_html=True
        )


        with st.spinner(
            "Generating customer-level explanation..."
        ):

            # ------------------------------------------------
            # GET PIPELINE COMPONENTS
            # ------------------------------------------------

            preprocessor = model.named_steps["preprocessor"]
            classifier = model.named_steps["classifier"]


            # ------------------------------------------------
            # TRANSFORM CUSTOMER DATA
            # ------------------------------------------------

            customer_transformed = (
                preprocessor.transform(customer_data)
            )


            # ------------------------------------------------
            # FEATURE NAMES
            # ------------------------------------------------

            feature_names = (
                preprocessor.get_feature_names_out()
            )


            # ------------------------------------------------
            # SHAP EXPLAINER
            # ------------------------------------------------

            explainer = shap.TreeExplainer(
                classifier
            )


            # ------------------------------------------------
            # CALCULATE SHAP
            # ------------------------------------------------

            shap_output = explainer.shap_values(
                customer_transformed
            )


            # ------------------------------------------------
            # HANDLE SHAP OUTPUT FORMAT
            # ------------------------------------------------

            if isinstance(
                shap_output,
                list
            ):

                # Older SHAP versions
                shap_customer = np.asarray(
                    shap_output[1]
                )[0]

            else:

                shap_array = np.asarray(
                    shap_output
                )

                if shap_array.ndim == 3:

                    # Shape: (samples, features, classes)
                    shap_customer = (
                        shap_array[0, :, 1]
                    )

                elif shap_array.ndim == 2:

                    shap_customer = (
                        shap_array[0]
                    )

                else:

                    shap_customer = (
                        shap_array.flatten()
                    )


            # ------------------------------------------------
            # GET VALUES
            # ------------------------------------------------

            if hasattr(
                customer_transformed,
                "toarray"
            ):

                customer_values = (
                    customer_transformed
                    .toarray()[0]
                )

            else:

                customer_values = (
                    np.asarray(
                        customer_transformed
                    )[0]
                )


            # ------------------------------------------------
            # CREATE SHAP DATAFRAME
            # ------------------------------------------------

            shap_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "SHAP Value": shap_customer,
                    "Feature Value": customer_values
                }
            )


            # ------------------------------------------------
            # SORT BY ABSOLUTE IMPACT
            # ------------------------------------------------

            shap_df["Absolute Impact"] = (
                shap_df["SHAP Value"].abs()
            )

            shap_df = (
                shap_df
                .sort_values(
                    "Absolute Impact",
                    ascending=False
                )
                .head(8)
                .sort_values(
                    "SHAP Value"
                )
            )


            # ------------------------------------------------
            # CLEAN FEATURE NAMES
            # ------------------------------------------------

            shap_df["Feature"] = (
                shap_df["Feature"]
                .str.replace(
                    "num__",
                    "",
                    regex=False
                )
                .str.replace(
                    "cat__",
                    "",
                    regex=False
                )
                .str.replace(
                    "_",
                    " ",
                    regex=False
                )
            )


            # ------------------------------------------------
            # CREATE CHART
            # ------------------------------------------------

            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            values = shap_df["SHAP Value"].values
            names = shap_df["Feature"].values


            bars = ax.barh(
                names,
                values
            )


            # Different visual appearance for
            # positive / negative SHAP values
            for bar, value in zip(
                bars,
                values
            ):

                if value >= 0:

                    bar.set_color(
                        "#ef4444"
                    )

                else:

                    bar.set_color(
                        "#22c55e"
                    )


            ax.axvline(
                0,
                linewidth=1
            )

            ax.set_xlabel(
                "SHAP Impact on Churn Prediction"
            )

            ax.set_title(
                "Top Factors Influencing This Customer"
            )

            ax.grid(
                axis="x",
                alpha=0.2
            )

            plt.tight_layout()


            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


            # ------------------------------------------------
            # LEGEND
            # ------------------------------------------------

            c1, c2 = st.columns(2)

            with c1:

                st.error(
                    "🔴 Positive SHAP → "
                    "pushes prediction toward churn"
                )

            with c2:

                st.success(
                    "🟢 Negative SHAP → "
                    "pushes prediction toward staying"
                )


            # ------------------------------------------------
            # TOP FACTORS TEXT
            # ------------------------------------------------

            positive_features = (
                shap_df[
                    shap_df["SHAP Value"] > 0
                ]
                .sort_values(
                    "SHAP Value",
                    ascending=False
                )
                .head(3)
            )


            negative_features = (
                shap_df[
                    shap_df["SHAP Value"] < 0
                ]
                .sort_values(
                    "SHAP Value"
                )
                .head(3)
            )


            if len(positive_features) > 0:

                st.markdown(
                    "### 🔴 Factors increasing churn risk"
                )

                for _, row in positive_features.iterrows():

                    st.write(
                        f"**{row['Feature']}** "
                        f"({row['SHAP Value']:+.3f})"
                    )


            if len(negative_features) > 0:

                st.markdown(
                    "### 🟢 Factors reducing churn risk"
                )

                for _, row in negative_features.iterrows():

                    st.write(
                        f"**{row['Feature']}** "
                        f"({row['SHAP Value']:+.3f})"
                    )


        # ====================================================
        # CUSTOMER SNAPSHOT
        # ====================================================

        st.markdown(
            """
<div class="section-title">
👤 Customer Snapshot
</div>

<div class="section-subtitle">
Key customer information used for this prediction.
</div>
""",
            unsafe_allow_html=True
        )


        c1, c2 = st.columns(2)


        with c1:

            st.markdown(
                f"""
<div class="info-box">

<b>Tenure:</b> {tenure} months
<br>

<b>Contract:</b> {contract}
<br>

<b>Internet Service:</b> {internet_service}
<br>

<b>Payment Method:</b> {payment_method}

</div>
""",
                unsafe_allow_html=True
            )


        with c2:

            st.markdown(
                f"""
<div class="info-box">

<b>Monthly Charges:</b> ₹{monthly_charges:.2f}
<br>

<b>Total Charges:</b> ₹{total_charges:.2f}
<br>

<b>Online Security:</b> {online_security}
<br>

<b>Tech Support:</b> {tech_support}

</div>
""",
                unsafe_allow_html=True
             )
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    """
<div class="card">

<div class="section-title">
🤖 Model Information
</div>

<div class="info-box">

<b>Algorithm:</b> Random Forest Classifier
<br><br>

<b>Prediction Threshold:</b> 40%
<br><br>

<b>Risk Levels:</b>
Low Risk &lt; 40% |
Medium Risk 40–69.9% |
High Risk ≥ 70%
<br><br>

<b>Explainability:</b>
SHAP-based customer-level feature attribution

</div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">

📊 Customer Churn Prediction
<br>

Machine Learning + SHAP Explainability
<br><br>

Built for predictive customer retention analytics.

</div>
""",
    unsafe_allow_html=True
)