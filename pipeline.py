# Локальный класс глубокого обучения
from deep_model import DeepModelTS
# Обработка данных
import pandas as pd
from datetime import datetime, timedelta
# Чтение файла конфигурации
import yaml
# Управление каталогом
import os
# графические пакеты
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Чтение параметров pipeline
with open(f'{os.getcwd()}\\conf.yml') as file:
    conf = yaml.load(file, Loader=yaml.FullLoader)

# Чтение данных
d = pd.read_csv('input/DAYTON_hourly.csv')
d['Datetime'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in d['Datetime']]

# Убедиться, что нет повторяющихся данных
# Если есть дубликаты, то усреднить данные за эти дублированные дни
d = d.groupby('Datetime', as_index=False)['DAYTON_MW'].mean()

# Сортировка значений
d.sort_values('Datetime', inplace=True)

# Инициализация класса глубокого обучения
deep_learner = DeepModelTS(
        data=d,
        Y_var='DAYTON_MW',
        lag=conf.get('lag'),
        LSTM_layer_depth=conf.get('LSTM_layer_depth'),
        epochs=conf.get('epochs'),
        # Доля данных, которые будут использоваться для проверки
        train_test_split=conf.get('train_test_split'),
)

# Получить модель
model = deep_learner.LSTModel()

# Прогноз на проверочном наборе
# Применимо, только если train_test_split в conf.yml> 0
yhat = deep_learner.predict()

if len(yhat) > 0:
    # Построение кадра данных прогноза
    fc = d.tail(len(yhat)).copy()
    fc.reset_index(inplace=True)
    fc['forecast'] = yhat

    # Составление прогнозов
    plt.figure(figsize=(20, 10))
    for dtype in ['DAYTON_MW', 'forecast']:
        plt.plot(
                'Datetime',
                dtype,
                data=fc,
                label=dtype,
                alpha=0.8
        )
    plt.legend()
    plt.grid()
    plt.show()

# Создание модели с использованием полных данных и прогнозирование на N шагов вперед
deep_learner = DeepModelTS(
        data=d,
        Y_var='DAYTON_MW',
        lag=24,
        LSTM_layer_depth=64,
        epochs=10,
        train_test_split=0
)

# Получить модель для прогнозирования на несколько шагов вперед
deep_learner.LSTModel()

# Прогнозирование на N шагов вперед
n_ahead = 168
yhat = deep_learner.predict_n_ahead(n_ahead)
yhat = [y[0][0] for y in yhat]

# Построение кадра данных прогноза
fc = d.tail(400).copy()
fc['type'] = 'original'

last_date = max(fc['Datetime'])
hat_frame = pd.DataFrame({
    'Datetime': [last_date + timedelta(hours=x + 1) for x in range(n_ahead)],
    'DAYTON_MW': yhat,
    'type': 'forecast'
})

fc = fc.append(hat_frame)
fc.reset_index(inplace=True, drop=True)

# Составление прогнозов
plt.figure(figsize=(20, 10))
for col_type in ['original', 'forecast']:
    plt.plot(
            'Datetime',
            'DAYTON_MW',
            data=fc[fc['type'] == col_type],
            label=col_type
    )

plt.legend()
plt.grid()
plt.show()
