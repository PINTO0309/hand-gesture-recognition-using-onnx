[Japanese / [WIP] English]

---
# hand-gesture-recognition-using-onnx
[Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe) This repository replaces the entire MediaPipe implementation with ONNX, borrowing from [@Kazuhito00](https://github.com/Kazuhito00). This is a sample program that recognizes hand signs and finger gestures using a simple MLP. The only thing I've confirmed so far is that it works.

https://user-images.githubusercontent.com/33194443/189632510-1823cd54-bc36-4889-ac14-adc16deba9b8.mp4

This repository contains the following content:
- Sample Program
- Hand Detection Model (Modified ONNX)
- Palm Landmark Detection Model (Modified ONNX)
- Hand Sign Recognition Model (Modified ONNX)
- Finger Gesture Recognition Model (Modified ONNX)
- Training Data and Training Notebook for Hand Sign Recognition
- Training Data and Training Notebook for Finger Gesture Recognition

# Requirements
- onnxruntime 1.12.0 or onnxruntime-gpu 1.12.0
- opencv-contrib-python 4.6.0.66 or Later
- Tensorflow 2.10.0 (Only if you want to recreate ONNX files after training)
- PyTorch 1.12.0 (Only if you want to recreate ONNX files after training)
- tf2onnx 1.12.0 or Later (Only if you want to recreate ONNX files after training)
- simple-onnx-processing-tools 1.0.54 or Later (Only if you want to recreate the ONNX file after training)
- scikit-learn 0.23.2 or Later (Only if you want to display the confusion matrix during training)
- matplotlib 3.3.2 or Later (Only if you want to display the confusion matrix during training)

# Demo
To run the demo using a webcam:
```bash
python app.py
```

The following options can be specified when running the demo.
```
--device
Specify the camera device number (default: 0)

--width
Width of camera capture (default: 640)

--height
Height of camera capture (default: 480)

--min_detection_confidence
Detection confidence threshold (default: 0.6)

--disable_image_flip
Disables horizontal flipping of input image
```

# Directory
```
.
│ app.py
│ keypoint_classification.ipynb
│ point_history_classification.ipynb
│ requirements.txt
│ README.md
│
├─model
│ ├─keypoint_classifier
│ │ │ tflite_to_onnx.sh
│ │ │ make_argmax.py
│ │ │ keypoint.csv
│ │ │ keypoint_classifier.hdf5
│ │ │ keypoint_classifier.py
│ │ │ keypoint_classifier.tflite
│ │ │ keypoint_classifier.onnx
│ │ └─ keypoint_classifier_label.csv
│ │
│ └─ point_history_classifier
│ │ tflite_to_onnx.sh
│ │ make_argmax.py
│ │ point_history.csv
│ │ point_history_classifier.hdf5
│ │ point_history_classifier.py
│ │ point_history_classifier.tflite
│ │ point_history_classifier.onnx
│ └─ point_history_classifier_label.csv
│
└─utils
│ cvfpscalc.py
└─ utils.py
```
### app.py
This is a sample program for inference. <br>You can also collect training data (keypoints) for hand sign recognition,<br>
and training data (index finger coordinate history) for finger gesture recognition.

### keypoint_classification.ipynb
Model training script for hand sign recognition.

### point_history_classification.ipynb
Model training script for finger gesture recognition.

### model/keypoint_classifier
This directory stores files related to hand sign recognition. <br>
The following files are stored here.
- tflite ONNX conversion script (tflite_to_onnx.sh)
- ONNX component generation program (make_argmax.py)
- Training data (keypoint.csv)
- Trained model (keypoint_classifier.tflite)
- Trained model (keypoint_classifier.onnx)
- Label data (keypoint_classifier_label.csv)
- Inference classes (keypoint_classifier.py)

### model/point_history_classifier
This directory contains files related to finger gesture recognition. <br>
The following files are stored here.
- tflite ONNX conversion script (tflite_to_onnx.sh)
- ONNX component generation program (make_argmax.py)
- Training data (point_history.csv)
- Trained model (point_history_classifier.tflite)
- Trained model (point_history_classifier.onnx)
- Label data (point_history_classifier_label.csv)
- Inference class (point_history_classifier.py)

### utils/cvfpscalc.py
Module for FPS measurement.

### utils/utils.py
Functions for image processing.

# Training
For hand sign recognition and finger gesture recognition, you can add or change training data and retrain the model.

### Hand Sign Recognition Training Method
#### 1. Training Data Collection
Pressing "k" will enter keypoint saving mode ("MODE: Logging Key Point" will be displayed).<br>
<img src="https://user-images.githubusercontent.com/37477845/102235423-aa6cb680-3f35-11eb-8ebd-5d823e211447.jpg" width="60%"><br><br>
Pressing "0" through "9" will add keypoints to "model/keypoint_classifier/keypoint.csv" as shown below. <br>
Column 1: Pressed number (used as class ID), Column 2: TrackID, Columns 3 and beyond: Keypoint coordinates<br>
<img src="https://user-images.githubusercontent.com/37477845/102345725-28d26280-3fe1-11eb-9eeb-8c938e3f625b.png" width="80%"><br><br>
Keypoint coordinates are saved after the following preprocessing steps have been performed up to ④. <br>
<img src="https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png" width="80%">
<img src="https://user-images.githubusercontent.com/37477845/102244114-418a3c00-3f3f-11eb-8eef-f658e5aa2d0d.png" width="80%"><br><br>
By default, three types of training data are included: paper (class ID: 0), rock (class ID: 1), and pointing hand (class ID: 2). <br>
Add classes from 3 onward as needed, or delete existing data from the CSV file to prepare your own training data. <br>
<img src="https://user-images.githubusercontent.com/37477845/102348846-d0519400-3fe5-11eb-8789-2e7daec65751.jpg" width="25%">　<img src="https://user-images.githubusercontent.com/37477845/102348855-d2b3ee00-3fe5-11eb-9c6d-b8924092a6d8.jpg" width="25%">　<img src="https://user-images.githubusercontent.com/37477845/102348861-d3e51b00-3fe5-11eb-8b07-adc08a48a760.jpg" width="25%">

#### 2. Model Training
Open "[keypoint_classification.ipynb](keypoint_classification.ipynb)" in Jupyter Notebook and run it in order from top to bottom. <br>
To change the number of classes in the training data, change the value of "NUM_CLASSES = 3" and modify the labels in "model/keypoint_classifier/keypoint_classifier_label.csv" as appropriate. <br><br>

#### X. Model Structure
The following is an image of the model provided in "[keypoint_classification.ipynb](keypoint_classification.ipynb)".
<img src="https://user-images.githubusercontent.com/37477845/102246723-69c76a00-3f42-11eb-8a4b-7c6b032b7e71.png" width="50%"
