#!/usr/bin/env python

import onnxruntime
import numpy as np
from typing import (
    Optional,
    List,
)


class PointHistoryClassifier(object):
    def __init__(
        self,
        model_path: Optional[str] = 'model/point_history_classifier/point_history_classifier.onnx',
        providers: Optional[List] = [
            # (
            #     'TensorrtExecutionProvider', {
            #         'trt_engine_cache_enable': True,
            #         'trt_engine_cache_path': '.',
            #         'trt_fp16_enable': True,
            #     }
            # ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
        score_th=0.5,
        invalid_value=0,
    ):
        """PointHistoryClassifier

        Parameters
        ----------
        model_path: Optional[str]
            ONNX file path for Palm Detection

        providers: Optional[List]
            Name of onnx execution providers
            Default:
            [
                (
                    'TensorrtExecutionProvider', {
                        'trt_engine_cache_enable': True,
                        'trt_engine_cache_path': '.',
                        'trt_fp16_enable': True,
                    }
                ),
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
        """
        # Model loading
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=providers,
        )
        self.providers = self.onnx_session.get_providers()

        self.input_shapes = [
            input.shape for input in self.onnx_session.get_inputs()
        ]
        self.input_names = [
            input.name for input in self.onnx_session.get_inputs()
        ]
        self.output_names = [
            output.name for output in self.onnx_session.get_outputs()
        ]
        self.score_th = score_th
        self.invalid_value = invalid_value


    def __call__(
        self,
        point_history: np.ndarray,
    ) -> np.ndarray:
        """PointHistoryClassifier

        Parameters
        ----------
        point_history: np.ndarray
            Landmarks [N, 32]

        Returns
        -------
        class_ids: np.ndarray
            int64[N]
            Index of Finger gesture
        """
        results = self.onnx_session.run(
            self.output_names,
            {input_name: point_history for input_name in self.input_names},
        )
        max_scores, class_ids = results
        invalid_idxs = max_scores < self.score_th
        class_ids[invalid_idxs] = self.invalid_value

        return class_ids
