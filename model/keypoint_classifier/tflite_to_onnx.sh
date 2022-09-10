#!/bin/bash

python -m tf2onnx.convert \
--opset 11 \
--tflite keypoint_classifier.tflite \
--output keypoint_classifier.onnx

onnxsim keypoint_classifier.onnx keypoint_classifier.onnx

sbi4onnx \
--input_onnx_file_path keypoint_classifier.onnx \
--output_onnx_file_path keypoint_classifier.onnx \
--initialization_character_string batch

sor4onnx \
--input_onnx_file_path keypoint_classifier.onnx \
--old_new "input_1" "input" \
--mode inputs \
--output_onnx_file_path keypoint_classifier.onnx

sor4onnx \
--input_onnx_file_path keypoint_classifier.onnx \
--old_new "Identity" "output" \
--mode outputs \
--output_onnx_file_path keypoint_classifier.onnx
