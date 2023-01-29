import numpy as np
import pandas as pd


# Метод создания матрицы X и Y из списка временных рядов
# для обучения модели глубокого обучения
def create_X_Y(ts: list, lag: int) -> tuple:
    X, Y = [], []

    if len(ts) - lag <= 0:
        X.append(ts)
    else:
        for i in range(len(ts) - lag):
            Y.append(ts[i + lag])
            X.append(ts[i:(i + lag)])

    X, Y = np.array(X), np.array(Y)

    # Преобразование массива X в форму ввода LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, Y


# Метод создания данных для модели нейронной сети
def create_data_for_NN(data: pd.DataFrame, Y_var: str, lag: int, use_last_n=None):
    # Извлечение основной переменной, которую хотим прогнозировать
    y = round(data[Y_var], 2).tolist()

    # Подстановка временных рядов при необходимости
    if use_last_n is not None:
        y = y[-use_last_n:]

    # Матрица X будет содержать N (например, lag = 3) измерений назад от значения Y
    # Если Y = [1434.0, 1489.0, 1620.0], то
    # X = [
    #     [1621.0, 1536.0, 1500.0],
    #     [1536.0, 1500.0, 1434.0],
    #     [1500.0, 1434.0, 1489.0],
    # ]
    # Первые три значения до 1434,0 (1500.0, 1536.0, 1621.0, 1434.0)
    X, Y = create_X_Y(y, lag)

    # Создание обучающих и тестовых наборов
    X_train = X
    X_test = []

    Y_train = Y
    Y_test = []

    return X_train, X_test, Y_train, Y_test


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
