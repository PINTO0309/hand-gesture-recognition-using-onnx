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

    def forward(self, scores):
        max_values, max_indices = torch.max(scores, dim=1)
        return max_indices


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
        default=3,
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

    torch.onnx.export(
        model,
        args=(scores),
        f=onnx_file,
        opset_version=OPSET,
        input_names=['argmax_input'],
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