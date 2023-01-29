# Обработка данных
import pandas as pd
import numpy as np

# Глубокое обучение
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.metrics import RootMeanSquaredError, Accuracy
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, Conv1D, MaxPooling1D, \
    Flatten, ConvLSTM2D
import tensorflow as tf

# Класс для создания модели глубокого обучения с использованием временного ряда
from keras.optimizers import Adam
from numpy import array


class DeepModelTS:
    def __init__(
            self,
            data: pd.DataFrame,
            Y_var: str,
            lag: int,
            LSTM_layer_depth: int,
            epochs=10,
            batch_size=256,
            train_test_split=0,
    ):

        self.model = None
        self.data = data
        self.Y_var = Y_var
        self.lag = lag
        self.LSTM_layer_depth = LSTM_layer_depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_test_split = train_test_split

    # Метод создания матрицы X и Y из списка временных рядов
    # для обучения модели глубокого обучения
    @staticmethod
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
    def create_data_for_NN(self, use_last_n=None):
        # Извлечение основной переменной, которую хотим прогнозировать
        y = self.data[self.Y_var].tolist()

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
        X, Y = self.create_X_Y(y, self.lag)

        # Создание обучающих и тестовых наборов
        X_train = X
        X_test = []

        Y_train = Y
        Y_test = []

        if self.train_test_split > 0:
            index = round(len(X) * self.train_test_split)

            X_train = X[:(len(X) - index)]
            X_test = X[-index:]

            Y_train = Y[:(len(X) - index)]
            Y_test = Y[-index:]

        return X_train, X_test, Y_train, Y_test

    # split a univariate sequence into samples
    def split_sequence(self, sequence, n_steps):
        X = list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence) - 1:
                break
            # gather input and output parts of the pattern
            seq_x = sequence[i:end_ix]
            X.append(seq_x)
        return array(X)

    # Метод создания модели LSTM
    def LSTModel(self):
        # Получение данных
        X_train, X_test, Y_train, Y_test = self.create_data_for_NN()
        # Для слоя Conv
        # X_train = X_train.reshape((X_train.shape[0], 1, self.lag, 1))

        # Определение модели
        model = Sequential()

        # ///////////////////////// MODEL #1 /////////////////////////////////////////////////
        # model.add(LSTM(64, activation='relu', input_shape=(self.lag, 1)))
        # model.add(Dense(1))

        # ///////////////////////// MODEL #2 /////////////////////////////////////////////////
        # model.add(LSTM(64, activation='relu', return_sequences=True,
        #                input_shape=(self.lag, 1)))
        # model.add(LSTM(48, activation='relu', return_sequences=True))
        # model.add(LSTM(20, activation='relu'))
        # model.add(Dense(1))

        # ///////////////////////// MODEL #3 /////////////////////////////////////////////////
        # model.add(Bidirectional(LSTM(64, activation='relu', input_shape=(self.lag, 1))))
        # model.add(Dense(1))

        # ///////////////////////// MODEL #4 /////////////////////////////////////////////////
        # model.add(Conv1D(filters=68, kernel_size=1, activation='relu',
        #                  input_shape=(self.lag, 1)))
        # model.add(LSTM(64, activation='relu'))
        # model.add(Dense(1))

        # ///////////////////////// MODEL #5 /////////////////////////////////////////////////
        # X_train = X_train.reshape((X_train.shape[0], 1, self.lag, 1))
        # model.add(TimeDistributed(Conv1D(filters=64,
        #                           kernel_size=1, activation='relu'),
        #                           input_shape=(None, self.lag, 1)))
        # model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        # model.add(TimeDistributed(Flatten()))
        # model.add(LSTM(64, activation='relu'))
        # model.add(Dense(1))

        # ///////////////////////// MODEL #6 /////////////////////////////////////////////////
        # X_train = X_train.reshape((X_train.shape[0], 1, self.lag, 1))
        # model.add(TimeDistributed(Conv1D(filters=256,
        #                           kernel_size=1, activation='relu'),
        #                           input_shape=(None, self.lag, 1)))
        # model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        # model.add(TimeDistributed(Flatten()))
        # model.add(Bidirectional(LSTM(128, activation='relu', return_sequences=True)))
        # model.add(LSTM(64, activation='relu'))
        # model.add(Dense(1))

        # ///////////////////////// MODEL #7 /////////////////////////////////////////////////
        X_train = X_train.reshape((X_train.shape[0], 1, 1, self.lag, 1))
        model.add(ConvLSTM2D(256,
                             kernel_size=(1, 2),
                             activation='relu',
                             input_shape=(1, 1, self.lag, 1)))
        model.add(Flatten())
        model.add(Dense(1))

        # Model #8
        # X_train = X_train.reshape((X_train.shape[0], self.lag, 1))
        # model.add(Conv1D(filters=64, kernel_size=3, activation='relu',
        #                  input_shape=(self.lag, 1)))
        # model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        # model.add(MaxPooling1D(pool_size=2))
        # model.add(Flatten())
        # model.add(RepeatVector(1))
        # model.add(LSTM(200, activation='relu', return_sequences=True))
        # model.add(TimeDistributed(Dense(100, activation='relu')))
        # model.add(TimeDistributed(Dense(1)))

        # Adding the output layer
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=0.001),
                      metrics=[RootMeanSquaredError(),
                               MeanSquaredError(),
                               MeanAbsoluteError(),
                               Accuracy()]
                      )

        # model.summary()

        # Определение параметра модели
        keras_dict = {
            'x': X_train,
            'y': Y_train,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'shuffle': False
        }

        if self.train_test_split > 0:
            keras_dict.update({
                'validation_data': (X_test, Y_test)
            })

        # Обучение модели
        model.fit(**keras_dict)

        # Сохранение модели в классе
        self.model = model

        return model

    # Метод прогнозирования с использованием тестовых данных,
    # использованных при создании класса
    def predict(self) -> list:
        yhat = []

        if self.train_test_split > 0:
            # Получение последних n временных рядов
            _, X_test, _, _ = self.create_data_for_NN()

            # Составление списка предсказаний
            yhat = [y[0] for y in self.model.predict(X_test)]

        return yhat

    # Метод прогнозирования N временных шагов вперед
    def predict_n_ahead(self, n_ahead: int):
        X, _, _, _ = self.create_data_for_NN(use_last_n=self.lag)

        # Составление списка предсказаний
        yhat = []

        for _ in range(n_ahead):
            # Прогноз
            # Для слоя Conv
            # Model  #5, #5.1, #5.2
            # X = X.reshape((1, 1, self.lag, 1))
            # Model #6
            X = X.reshape((1, 1, 1, self.lag, 1))

            fc = self.model.predict(X)
            yhat.append(fc)

            # Создание новой входной матрицы для прогнозирования
            X = np.append(X, fc)
            # Исключение первой переменной
            X = np.delete(X, 0)
            # Изменение формы для следующей итерации
            X = np.reshape(X, (1, len(X), 1))

        return yhat

    # Для обеспечения воспроизводимости результатов на CPU
    # tf.random.set_seed(13)
    print(tf.config.list_physical_devices('GPU'))
