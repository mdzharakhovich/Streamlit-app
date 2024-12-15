#your code here

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)

# все, которые еще сочтете нужными
#your code here
data = pd.read_csv("data.adult.csv")

data_clean = data.dropna()
data_clean.isnull().sum()

# целевая переменная доход
income = data_clean['>50K,<=50K']
data_clean.drop(columns = ['>50K,<=50K'])
income = income.map({'>50K': 1, '<=50K': 0})
num_data = data_clean.select_dtypes(include=['float64', 'int64'])
X_train, X_test, y_train, y_test = train_test_split(num_data, income, test_size=0.2, random_state=42)


# начинаем строить модели
# Определение параметров для GridSearchCV
param_grid = {
    'max_depth': [None, 3, 5, 10, 15, 20]  # Возможные значения max_depth
}

# Определение схемы кросс-валидации
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Список моделей для подбора гиперпараметров
models = [
    ('DecisionTree', DecisionTreeClassifier(random_state=42)),
    ('RandomForest', RandomForestClassifier(random_state=42)),
    ('GradientBoosting', GradientBoostingClassifier(random_state=42))
]

# Подбор гиперпараметров для каждой модели
for model_name, model in models:
    print(f"Подбор гиперпараметров для {model_name}:")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Вывод лучших параметров и лучшего результата
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучшее значение accuracy на кросс-валидации: {grid_search.best_score_:.4f}")
    
    # Оценка модели на тестовой выборке
    y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy на тестовой выборке: {test_accuracy:.4f}")
    print("-" * 50)



