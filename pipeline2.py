# Локальный класс глубокого обучения
from deep_model import DeepModelTS
# Обработка данных
import pandas as pd
# Чтение файла конфигурации
import yaml
# Управление каталогом
import os
# Графические пакеты
import matplotlib.pyplot as plt

# WebAgg/Qt5Agg
plt.matplotlib.use('WebAgg')
# Работа с датой и временим
import time
from datetime import datetime

# Чтение параметров pipeline
with open(f'{os.getcwd()}\\conf.yml') as file:
    conf = yaml.load(file, Loader=yaml.FullLoader)

templite_date = '%Y-%m-%d %H:%M:%S'

# Чтение данных
d_all = pd.read_csv(conf.get('file'))
d_all[conf.get('X_var')] = [datetime.strptime(x, templite_date) for x in
                            d_all[conf.get('X_var')]]

# Прогнозирование на N шагов вперед
n_ahead = 72
n_days = 1
# 96449 (2012)
# 140438 (2017)
# 175612 (2021)
start_data = 175612

d = d_all[start_data:len(d_all) - n_ahead * n_days]
true_value = d_all[len(d_all) - n_ahead * n_days:][conf.get('Y_var')].tolist()

# Прогнозирование на N шагов вперед
result = []
yhat = []

step_time = 3600

for _ in range(0, n_days):
    # Создание модели с использованием полных данных и прогнозирование на N шагов вперед
    deep_learner = DeepModelTS(
            data=d,
            Y_var=conf.get('Y_var'),
            lag=conf.get('lag'),
            LSTM_layer_depth=conf.get('LSTM_layer_depth'),
            batch_size=conf.get('batch_size'),
            epochs=conf.get('epochs'),
            train_test_split=0
    )

    # Получить модель для прогнозирования на несколько шагов вперед
    deep_learner.LSTModel()

    yhat = deep_learner.predict_n_ahead(n_ahead)
    yhat = [round(y[0][0], 2) for y in yhat]

    for y in yhat:
        result.append(y)

    end_date_str = str(d.tail(1).get(conf.get('X_var')).to_string()).split('   ')[1]
    if len(end_date_str) == 10:
        end_ts_add_1h = time.mktime(datetime.strptime(end_date_str + ' 00:00:00',
                                                      '%Y-%m-%d %H:%M:%S').timetuple())
    else:
        end_ts = time.mktime(datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S').timetuple())
        end_ts_add_1h = end_ts + step_time

    res = pd.date_range(
            datetime.strptime(
                    datetime.fromtimestamp(end_ts_add_1h).strftime('%Y-%m-%d %H:%M:%S'),
                    '%Y-%m-%d %H:%M:%S'),
            periods=n_ahead,
            freq='H')

    df = res.to_frame(index=False, name=conf.get('X_var'))
    df[conf.get('Y_var')] = yhat
    frames = [d, df]

    d = pd.concat(frames, ignore_index=True)

# Absolute error
a = []
for i in range(0, (n_ahead * n_days)):
    a.append(round(abs(true_value[i] - result[i]), 2))

absolute_error = round(sum(a) / (n_ahead * n_days), 2)

# Relative errors
e = []
for i in range(0, (n_ahead * n_days)):
    e.append(round(abs(true_value[i] - result[i]) / true_value[i] * 100, 2))

relative_error = round(abs(sum(e) / (n_ahead * n_days)))

# Составление прогнозов (график)
plt.figure(figsize=(15, 6))
plt.plot(true_value, c="blue", linewidth=3, label='Реальные значения',
         marker='o')

for x, y in zip(true_value[0:len(true_value):10],
                range(0, (n_ahead * n_days))[0:n_ahead * n_days:10]):
    plt.text(y + .8, x + .2, str(x), color='blue')

plt.plot(result, c="green", linewidth=3, label='Прогнозируемые значения', marker='X')
for x, y in zip(result[0:len(result):10], range(0, (n_ahead * n_days))[0:n_ahead * n_days:10]):
    plt.text(y + .8, x + .2, str(x), color='green')

plt.title('Модель №7 - ConvLSTM2D')
plt.xticks(range(0, (n_ahead * n_days)))
plt.xlabel(f'Часы\n'
           f'Абсолютная ошибка: {absolute_error}, относительная ошибка: {relative_error}%')
plt.ylabel('Погода')

plt.grid()
plt.legend()

plt.show()
