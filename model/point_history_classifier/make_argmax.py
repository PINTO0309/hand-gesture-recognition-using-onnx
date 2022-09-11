#! /usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnxsim import simplify
from argparse import ArgumentParser

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.invalid_value = 0

    def forward(self, scores, score_threshold):
        max_scores, class_ids = torch.max(scores, dim=1)

        invalid_idxs = max_scores < score_threshold
        class_ids[invalid_idxs] = self.invalid_value

        return class_ids


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-o',
        '--opset',
        type=int,
        default=11,
        help='onnx opset'
    )
    parser.add_argument(
        '-b',
        '--batches',
        type=int,
        default=1,
        help='batch size'
    )
    parser.add_argument(
        '-c',
        '--classes',
        type=int,
        default=4,
        help='classes'
    )
    args = parser.parse_args()

    model = Model()

    MODEL = f'argmax'
    OPSET=args.opset
    BATCHES = args.batches
    CLASSES = args.classes

    onnx_file = f"{MODEL}.onnx"
    scores = torch.randn(BATCHES, CLASSES)
    score_threshold = torch.tensor(0.5, dtype=torch.float32)

    torch.onnx.export(
        model,
        args=(scores, score_threshold),
        f=onnx_file,
        opset_version=OPSET,
        input_names=['argmax_input', 'score_threshold'],
        output_names=['class_ids'],
        dynamic_axes={
            'argmax_input' : {0: 'batch'},
            'class_ids' : {0: 'batch'},
        },
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)