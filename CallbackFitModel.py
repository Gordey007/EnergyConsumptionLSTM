import tensorflow as tf

ACCURACY_THRESHOLD = 50


class CallbackFitModel(tf.keras.callbacks.Callback):
    def __init__(self, point):
        super(CallbackFitModel, self).__init__()
        self.point = point

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs["mean_absolute_error"]
        if accuracy <= self.point:
            self.model.stop_training = True

    # def on_epoch_end(self, epoch, logs=None):
    #     if logs.get('mean_absolute_error') <= ACCURACY_THRESHOLD:
    #         print("\nReached %2.2f%% accuracy, so stopping training!!"
    #               % (ACCURACY_THRESHOLD * 100))
    #         self.model.stop_training = True

