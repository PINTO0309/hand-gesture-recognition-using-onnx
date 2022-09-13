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
        model_path: Optional[str] = 'model/point_history_classifier/point_history_classifier_lstm.onnx',
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
        self.score_th = np.asarray(score_th, dtype=np.float32)


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
            ClassIDs of Finger gesture
        """
        class_ids = self.onnx_session.run(
            self.output_names,
            {
                self.input_names[0]: point_history,
                self.input_names[1]: self.score_th,
            },
        )[0]

        return class_ids
