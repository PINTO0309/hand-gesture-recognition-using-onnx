#!/bin/bash

if [ $# -gt 1 ]; then
    echo "The number of arguments specified is $#." 1>&2
    echo "Be sure to specify 0 or 1 (Number of classes) argument. (default:4)" 1>&2
    exit 1
fi

if [ $# -eq 0 ]; then
    CLASSES=4
else
    CLASSES=$1
fi

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
--old_new "Identity" "base_scores" \
--mode outputs \
--output_onnx_file_path point_history_classifier.onnx

python make_argmax.py --classes ${CLASSES}

snc4onnx \
-if point_history_classifier.onnx argmax.onnx \
-of point_history_classifier.onnx \
-sd base_scores argmax_input

rm argmax.onnx
