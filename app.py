import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Заголовок приложения
st.title("Прогнозирование заработка")
st.write("Это приложение предсказывает, превысит ли ваш средний заработок порог $50k.")

# Загрузка модели
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

relationship = st.sidebar.selectbox("Отношения", relationship_options)
race = st.sidebar.selectbox("Раса", race_options)
sex = st.sidebar.selectbox("Пол", sex_options)

# Создание DataFrame из введенных данных
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

# Преобразование категориальных признаков в one-hot encoded формат
input_data_encoded = pd.get_dummies(input_data, columns=['relationship', 'race', 'sex'], drop_first=False)

# Убедитесь, что порядок столбцов в input_data_encoded совпадает с порядком столбцов, использованным при обучении модели
# Получите порядок столбцов, использованных при обучении модели
expected_columns = model.feature_names_in_

# Добавьте отсутствующие столбцы с нулевыми значениями
missing_columns = set(expected_columns) - set(input_data_encoded.columns)
for column in missing_columns:
    input_data_encoded[column] = 0

# Упорядочиваем столбцы в input_data_encoded в том же порядке, что и при обучении модели
input_data_encoded = input_data_encoded[expected_columns]

# Прогнозирование
if st.button("Предсказать"):
    prediction = model.predict(input_data_encoded)
    if prediction[0] == 1:
        st.success("Ваш заработок превысит $50k!")
    else:
        st.error("Ваш заработок НЕ превысит $50k.")