import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Заголовок приложения
st.title("Прогнозирование заработка")
st.write("Это приложение предсказывает, превысит ли ваш средний заработок порог $50k.")

# Загрузка модели
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Ввод данных пользователем
st.sidebar.header("Введите данные")
age = st.sidebar.slider("Возраст", 17, 90, 30)
education_num = st.sidebar.slider("Образование (education-num)", 1, 16, 10)
hours_per_week = st.sidebar.slider("Часов в неделю", 1, 99, 40)
capital_gain = st.sidebar.number_input("Капитальный доход", 0, 99999, 0)
capital_loss = st.sidebar.number_input("Капитальный убыток", 0, 4356, 0)

# Преобразование данных в DataFrame
data = pd.DataFrame({
    'age': [age],
    'education-num': [education_num],
    'hours-per-week': [hours_per_week],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss]
})

# Кнопка для предсказания
if st.button("Предсказать"):
    prediction = model.predict(data)
    if prediction[0] == 1:
        st.success("Ваш заработок превысит $50k!")
    else:
        st.error("Ваш заработок НЕ превысит $50k.")


# сюда добавить лучшую модель 
# best_model = 

#import pickle
#with open('best_model.pkl', 'wb') as file:
  #  pickle.dump(best_model, file)