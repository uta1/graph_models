from lib_imports import *
import tensorflow as tf


class ObjectDetectionRate(tf.keras.metrics.Metric):
    def __init__(self, name='object_detection_rate', num_classes=6, **kwargs):
        super(ObjectDetectionRate, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.reset_states()

    def update_state(self, y_pred, y_true, anns):
        pass

    def result(self):
        return 0.0

    def reset_states(self):
        self.tp = np.array(self.num_classes, dtype=np.int32)
        self.fp = np.array(self.num_classes, dtype=np.int32)
        self.tn = np.array(self.num_classes, dtype=np.int32)
        self.fn = np.array(self.num_classes, dtype=np.int32)
