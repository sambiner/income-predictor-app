import streamlit as st
import pickle
import numpy as np
import pandas as pd


@st.cache_resource
def load_model():
    with open("model_V2.pkl", "rb") as m:
        model = pickle.load(m)
    return model


@st.cache_resource
def load_encoder():
    with open("encoder_V2.pkl", "rb") as e:
        encoder_dict = pickle.load(e)
    return encoder_dict


model = load_model()
encoder_dict = load_encoder()

numeric_features = ["age", "capitalgain", "capitalloss", "hoursperweek"]
categorical_features = [
    "workclass",
    "education",
    "maritalstatus",
    "occupation",
    "relationship",
    "race",
    "gender",
    "nativecountry",
]
feature_order = [
    "age",
    "workclass",
    "education",
    "maritalstatus",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capitalgain",
    "capitalloss",
    "hoursperweek",
    "nativecountry",
]

st.title("Income Predictor App")
st.write(
    "Enter your information below to predict whether your income will be above or below a threshold!"
)

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    capitalgain = st.number_input("Capital Gain", min_value=0, value=0)
    capitalloss = st.number_input("Capital Loss", min_value=0, value=0)
    hoursperweek = st.number_input(
        "Hours per Week", min_value=1, max_value=120, value=40
    )

    st.write("### Categorical Factors:")
    workclass = st.selectbox("Workclass", encoder_dict["workclass"])
    education = st.selectbox("Education", encoder_dict["education"])
    maritalstatus = st.selectbox("Marital Status", encoder_dict["maritalstatus"])
    occupation = st.selectbox("Occupation", encoder_dict["occupation"])
    relationship = st.selectbox("Relationship", encoder_dict["relationship"])
    race = st.selectbox("Race", encoder_dict["race"])
    gender = st.selectbox("Gender", encoder_dict["gender"])
    nativecountry = st.selectbox("Native Country", encoder_dict["nativecountry"])

    submitted = st.form_submit_button("Predict Your Income!")

if submitted:
    input_data = {
        "age": age,
        "workclass": workclass,
        "education": education,
        "maritalstatus": maritalstatus,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "gender": gender,
        "capitalgain": capitalgain,
        "capitalloss": capitalloss,
        "hoursperweek": hoursperweek,
        "nativecountry": nativecountry,
    }

    input_df = pd.DataFrame([input_data])

    # Use the encoder to convert strings to labels by mimicing the LabelEncoder
    for col in categorical_features:
        classes = encoder_dict[col]
        value = input_df.loc[0, col]
        if value not in classes:
            value = "Unknown"
        input_df.loc[0, col] = classes.index(value)

    for col in numeric_features:
        input_df[col] = pd.to_numeric(input_df[col])

    input_df = input_df[feature_order]

    st.write("### Input Data")
    st.dataframe(input_df)

    # Make the prediction
    prediction = model.predict(input_df)[0]
    income_classes = encoder_dict["income"]
    predicted_income = income_classes[prediction]

    st.write("### Prediction:")
    st.write(f"Predicted Income - **{predicted_income}**")
