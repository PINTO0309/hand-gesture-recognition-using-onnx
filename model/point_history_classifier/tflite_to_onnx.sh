#!/bin/bash

python -m tf2onnx.convert \
--opset 11 \
--tflite point_history_classifier.tflite \
--output point_history_classifier.onnx

onnxsim point_history_classifier.onnx point_history_classifier.onnx

sbi4onnx \
--input_onnx_file_path point_history_classifier.onnx \
--output_onnx_file_path point_history_classifier.onnx \
--initialization_character_string batch

sor4onnx \
--input_onnx_file_path point_history_classifier.onnx \
--old_new "input_1" "input" \
--mode inputs \
--output_onnx_file_path point_history_classifier.onnx

sor4onnx \
--input_onnx_file_path point_history_classifier.onnx \
--old_new "Identity" "output" \
--mode outputs \
--output_onnx_file_path point_history_classifier.onnx
