import streamlit as st
import pandas as pd
import joblib

model = joblib.load("LogisticRegression_heart.pkl")
scaler = joblib.load("heart_scaler1.pkl")
expected_columns = joblib.load("heart_columns.pkl")

st.title("Heart stroke prediction by Shaeeb ❤️")
st.markdown("Provide the following details")
age = st.slider("Age",18,100,40) # 18 to 100
sex = st.selectbox("SEX",['M','F'])
chest_pain = st.selectbox("Chest pain Type",['ATA','NAP','TA','ASY'])
resting_BP= st.number_input('Resting Blood Pressure(mm hg)',80,200,120)
cholesterol = st.number_input("Cholestrol (mg/dL)",100,600,200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dL',[0,1])
resting_ecg=st.selectbox("Resting ECG",['Normal',"ST","LVH"])
max_hr = st.slider("max Heart Rate",60,220,150)
exercise_angina = st.selectbox("Exercise=Induced Angina",['Y','N'])
oldpeak = st.slider("Oldpeak(ST Depression)",0.0,6.0,1.0)
st_slope = st.selectbox("ST Slope",['Up','Flat','Down'])

if st.button('Predict'):
    raw_input={
         'Age':age,
         'Sex_'+sex : 1,
         'ChestPainType_' + chest_pain: 1,
         'RestingBP': resting_BP,
         'Cholesterol': cholesterol,
         'FastingBS':fasting_bs,
         'Oldpeak': oldpeak,
         'RestingECG'+ resting_ecg: 1,
         'ExerciseAngina' + exercise_angina: 1,
         'ST_Slope' + st_slope: 1

    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0
    
    input_df = input_df[expected_columns]
    
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaler)[0] # 0th column select

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")