# NudeNet: lightweight Nudity detection

https://nudenet.notai.tech/ in-browser demo (the detector is run client side, i.e: in your browser, images are not sent to a server)

```bash
pip install --upgrade nudenet
```

```python
from nudenet import NudeDetector
nude_detector = NudeDetector()
nude_detector.detect('image.jpg') # Returns list of detections
```

```python
detection_example = [
 {'class': 'BELLY_EXPOSED',
  'score': 0.799403190612793,
  'box': [64, 182, 49, 51]},
 {'class': 'FACE_FEMALE',
  'score': 0.7881264686584473,
  'box': [82, 66, 36, 43]},
 ]
```

```python
all_labels = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]
```


### Docker

```bash
docker run -it -p8080:8080 ghcr.io/notai-tech/nudenet:latest
```

```bash
curl -F f1=@"images.jpeg" "http://localhost:8080/infer"

{"prediction": [[{"class": "BELLY_EXPOSED", "score": 0.8511635065078735, "box": [71, 182, 31, 50]}, {"class": "FACE_FEMALE", "score": 0.8033977150917053, "box": [83, 69, 21, 37]}, {"class": "FEMALE_BREAST_EXPOSED", "score": 0.7963727712631226, "box": [85, 137, 24, 38]}, {"class": "FEMALE_BREAST_EXPOSED", "score": 0.7709134817123413, "box": [63, 136, 20, 37]}, {"class": "ARMPITS_EXPOSED", "score": 0.7005534172058105, "box": [60, 127, 10, 20]}, {"class": "FEMALE_GENITALIA_EXPOSED", "score": 0.6804671287536621, "box": [81, 241, 14, 24]}]], "success": true}⏎
```

# **NUDENET Torch2TRT-batchedNMS**

## Environment

I'm running with docker `nvcr.io/nvidia/tritonserver:22.12`

- Python 3.8
- Torch 1.13.1
- ONNX 1.14.0
- Tensorrt 8.5.1.7

## Convert Nude Pytorch to ONNX with Post-Processing
- Open file ```onnx_post_process.py``` and update attribute values to suit your model
- Run: 
```Shell
CUDA_VISIBLE_DEVICES=1 python onnx_post_process.py --weights nudenet/best.onnx --output <your_output_model_name>.onnx
```
## Add NMS Batched to onnx model
- Open file ```add_nms_plugins.py``` and update attribute values to suit your model
- Run:
```Shell
python3 add_nms_plugins.py --model <your_output_model_name>.onnx
```
## Convert ONNX model to TrT model
- Run:
```Shell
/usr/src/tensorrt/bin/trtexec --onnx=<your_output_model_name>-nms.onnx \
                                --saveEngine=<your_output_trt_model_name>.trt \
                                --explicitBatch \
                                --minShapes=images:1x3x320x320 \
                                --optShapes=images:1x3x320x320 \
                                --maxShapes=images:1x3x320x320 \
                                --verbose \
                                --device=1
```

## Inference
- Open file ```object_detector_trt_nms.py``` and modify attribute values
- Run: 
```Shell
python3 object_detector_trt_nms.py
```

# REFERENCE
1. https://github.com/ultralytics/ultralytics

