#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class CNNKeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/cnn_keypoint_classifier/cnn_keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        """
        Classify hand gesture from keypoint landmarks.

        Args:
            landmark_list: List of 42 normalized keypoint values
                           (21 keypoints × 2 coordinates: x, y)

        Returns:
            int: Predicted class index
        """
        input_details_tensor_index = self.input_details[0]['index']
        # Reshape from (42,) to (21, 2) for CNN input
        landmark_array = np.array(landmark_list, dtype=np.float32).reshape(21, 2)
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_array], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index
