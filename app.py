import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Гадание на зарплату")
st.write("Это приложение предсказывает, превысит ли ваш заработок ПЯТЬДЕСЯТ. ТЫСЯЧ. ДОЛЛАРОВ (голосом Дудя*).")

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Ввод данных пользователем
st.sidebar.header("Введите данные")
age = st.sidebar.slider("Возраст", 18, 100, 30)
fnlwgt = st.sidebar.slider("Final Weight (fnlwgt)", 12285, 1484705, 100000)
education_num = st.sidebar.slider("Образование (число)", 1, 16, 10)
capital_gain = st.sidebar.slider("Капитальный прирост", 0, 100000, 0)
capital_loss = st.sidebar.slider("Капитальный убыток", 0, 10000, 0)
hours_per_week = st.sidebar.slider("Часов в неделю", 1, 100, 40)

# Категориальные признаки
relationship_options = ['Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife']
race_options = ['Asian-Pac-Islander', 'Black', 'Other', 'White']
sex_options = ['Male', 'Female']

relationship = st.sidebar.selectbox("Семейное положение", relationship_options)
race = st.sidebar.selectbox("Раса", race_options)
sex = st.sidebar.selectbox("Пол", sex_options)

input_data = pd.DataFrame({
    'age': [age],
    'fnlwgt': [fnlwgt],
    'education-num': [education_num],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'relationship': [relationship],
    'race': [race],
    'sex': [sex]
})

input_data_encoded = pd.get_dummies(input_data, columns=['relationship', 'race', 'sex'], drop_first=False)


expected_columns = model.feature_names_in_

missing_columns = set(expected_columns) - set(input_data_encoded.columns)
for column in missing_columns:
    input_data_encoded[column] = 0

input_data_encoded = input_data_encoded[expected_columns]

# Прогнозирование
if st.button("Предсказать"):
    prediction = model.predict(input_data_encoded)
    if prediction[0] == 1:
        st.success("Ваш заработок превысит $50k!")
        st.image("da8775f2-f63a-4dbe-89f5-31d8ba85a38d-2568619454.png", caption="Поздравляем! *Если что, Дудь -- иноагент.", use_container_width=True)
           # Добавляем бомбический трек для положительного прогноза
        st.audio("lida-serega-pirat-chsv (mp3cut.net).mp3", format="audio/mp3")
    else:
        st.error("Ваш заработок НЕ превысит $50k.")
        st.image("Sad-Pepe-The-Frog-PNG-Transparent-Picture-4087480041.png", caption="Нам очень жаль...  *Если что, Дудь -- иноагент.", use_container_width=True)