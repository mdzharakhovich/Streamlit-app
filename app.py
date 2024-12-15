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
education_num = st.sidebar.slider("Образование (число)", 1, 16, 10)
capital_gain = st.sidebar.slider("Капитальный прирост", 0, 100000, 0)
capital_loss = st.sidebar.slider("Капитальный убыток", 0, 10000, 0)
hours_per_week = st.sidebar.slider("Часов в неделю", 1, 100, 40)

# Создание DataFrame из введенных данных
input_data = pd.DataFrame({
    'age': [age],
    'education-num': [education_num],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week]
})

# Прогнозирование
prediction = model.predict(input_data)

# Кнопка для предсказания
if st.button("Предсказать"):
    prediction = model.predict(data)
    if prediction[0] == 1:
        st.success("Ваш заработок превысит $50k!")
    else:
        st.error("Ваш заработок НЕ превысит $50k.")