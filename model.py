import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv("data.adult.csv")

data.replace("?", np.nan, inplace=True)
data_clean = data.dropna()

income = data_clean['>50K,<=50K'].map({'>50K': 1, '<=50K': 0})
num_data = data_clean.select_dtypes(include=['float64', 'int64'])

X_train, X_test, y_train, y_test = train_test_split(num_data, income, test_size=0.2, random_state=42)

# Подбор гиперпараметров 
param_grid = {
    'max_depth': [3, 5, 10, 15],
    'n_estimators': [50, 100, 150, 200, 300]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

gb_model = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Лучшие параметры: {grid_search.best_params_}")
print(f"Лучшее значение accuracy на кросс-валидации: {grid_search.best_score_:.4f}")

# Оценка на тестовой выборке
y_pred = grid_search.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy на тестовой выборке: {test_accuracy:.4f}")

# Сохранение модели в pickle
with open('best_model.pkl', 'wb') as file:
    pickle.dump(grid_search.best_estimator_, file)
print("Модель GradientBoosting успешно сохранена в файл best_model.pkl")