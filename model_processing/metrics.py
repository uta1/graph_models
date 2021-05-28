from lib_imports import *
import tensorflow as tf

from logger import logger
from utils.geometry import rects_intersection


class ObjectDetectionRate(tf.keras.metrics.Metric):
    def __init__(self, name='object_detection_rate', num_classes=6, iou_threshold=0.5, **kwargs):
        super(ObjectDetectionRate, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset_states()

    def update_state(self, y_pred, anns, rois):
        was_detected = [False] * len(anns)
        for image_y_pred, roi in zip(y_pred, rois):
            if image_y_pred == 0:
                continue
            roi_square = roi[-2] * roi[-1]
            detected = False
            for ind, ann in enumerate(anns):
                if image_y_pred == ann['category_id']:
                    intersection_square = rects_intersection(ann['bbox'], roi)
                    iou_score = float(intersection_square) / (roi_square + ann['bbox'][-2] * ann['bbox'][-1] - intersection_square)
                    if iou_score >= self.iou_threshold:
                        detected = True
                        was_detected[ind] = True
                        break
            if detected:
                self.true_positives[image_y_pred] += 1
            else:
                self.false_positives[image_y_pred] += 1
        for dataset_object_detected, ann in zip(was_detected, anns):
            self.total_in_dataset[ann['category_id']] += 1
            if not dataset_object_detected:
                continue
            self.dataset_objects_detected[ann['category_id']] += 1

    @tf.autograph.experimental.do_not_convert
    def result(self):
        logger.log('true_positives: ' + str(self.true_positives[1:]))
        logger.log('false_positives: ' + str(self.false_positives[1:]))
        logger.log('total_in_dataset: ' + str(self.total_in_dataset[1:]))
        logger.log('dataset_objects_detected: ' + str(self.dataset_objects_detected[1:]))

        recall = self.true_positives[1:] / self.total_in_dataset[1:]
        precision = self.true_positives[1:] / (self.true_positives[1:] + self.false_positives[1:] + 0.0001)
        logger.log('recall: ' + str(recall))
        logger.log('precision: ' + str(precision))

        return recall * precision / (recall + precision + 0.0001)

    def reset_states(self):
        self.true_positives = np.zeros((self.num_classes,), dtype=np.float)
        self.false_positives = np.zeros((self.num_classes,), dtype=np.float)
        self.total_in_dataset = np.zeros((self.num_classes,), dtype=np.float)
        self.dataset_objects_detected = np.zeros((self.num_classes,), dtype=np.float)
