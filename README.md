# Описание решения

## Подготовка данных
- Для обучения модели использовались данные о погоде из файлов `data/train.csv` и `data/test.csv`.#
- Файл edit_features.csv - результат добавления признаков из open_data.csv, которые получены по api meteostats
- Использовался алгоритм ближайших соседей для поиска ближайших точек с данными о погоде для каждой точки из тренировочного и тестового наборов.
- Данные о погоде были объединены с тренировочным и тестовым наборами данных.
- Использован KMeans для добавления новых признаков исходя из анализа sns.pairplot()
- Использован CatBoostRegressor.select_features() для определения наиболее релевантных признаков для дальнейшего обучения модели
- Ввиду отстутствия score в тестовой выборке оценка проводится по среднему для mae на тестовой выборке, в результате использования кластеризации и select_features() среднее по mae для 5 фолдов изменилось следующим образом: 0.0586 -> 0.057

## Обучение модели

- Для обучения модели применялся градиентный бустинг с помощью CatBoostRegressor.
- Использовалась кросс-валидация с пятью фолдами для оценки качества модели на тренировочных данных.

## Прогнозирование

- Обученная модель была загружена из файла `catboost_model.bin`.
- Для тестового набора данных были сделаны прогнозы с использованием полученных данных о погоде.

## Подготовка файла с результатами

- Результаты прогнозирования были сохранены в файл `submission.csv`, где каждая строка содержит идентификатор точки и прогнозируемый показатель.
