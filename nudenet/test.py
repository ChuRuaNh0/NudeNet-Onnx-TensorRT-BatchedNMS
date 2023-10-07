import cv2
import os
import numpy as np
import onnxruntime
# import onnxruntime as rt
import copy.deepcopy as deepcopy
onnxruntime.get_device()
from utils_file import NudeDetector, \
                    FlagDetectionYolov5, \
                    DashLineDetectionYolov8, \
                    GoreHorrorClassify, \
                    upload_url_from_frame_image,\
                    upload_data_each_frame

# ================================MODEL============================================
nudenet_model = NudeDetector(weight="weights/nudenet_best.onnx")
dashline_model = DashLineDetectionYolov8(weights="weights/9-dashline-v4.pt", device="cuda:0", img_size=640)
flag_model = FlagDetectionYolov5(weights="weights/flag_detection_v3.pt", device="0", img_size=640)
gore_model = GoreHorrorClassify(weight="weights/model_gore.onnx")
horror_model = GoreHorrorClassify(weight="weights/model_horror.onnx")

# =================================id unique of model==================================
PGIE_FLAG_DETECTION_ID = 1
PGIE_DASHLINE_DETECTION_ID = 2
PGIE_SEX_CLASSIFICATION_ID = 3
PGIE_GORE_CLASSIFICATION_ID = 4
PGIE_HORROR_CLASSIFICATION_ID = 5
PGIE_VIOLENCE_CLASSIFICATION_ID = 6
PGIE_HUMAN_DETECTION_ID = 7
PGIE_NUDE_NET_DETECTION_ID = 8

# ================================DEFINE PARAMS====================================

ID_NUDE_NET_COVERED = [0, 10, 15, 16, 17]
ID_NUDE_NET_EXPOSED = [2, 3, 4, 6, 14]
ID_NUDE_NET_ALL = [0, 10, 15, 16, 17, 2, 3, 4, 6, 14]

ID_VIOLATES = {PGIE_FLAG_DETECTION_ID: ["4bcd8d0b-f8d7-4531-90d5-d8b23ce1a14c", "983e8a87-f511-41cd-83fc-e342cae3d75e", "e8286c99-5523-49fa-ab6f-2e41a1139771", "", ""],
               PGIE_DASHLINE_DETECTION_ID: ["48f6b3b4-edd7-4729-a179-d373947b2797"],
               PGIE_SEX_CLASSIFICATION_ID: ["32834c32-fe64-461b-b925-da8443836b87", "1b15d3d1-d111-4cdb-8c07-bcfa90f5ba78", "", "1b15d3d1-d111-4cdb-8c07-bcfa90f5ba78", "1b15d3d1-d111-4cdb-8c07-bcfa90f5ba78"],
               PGIE_GORE_CLASSIFICATION_ID: ["03e1e976-b9cc-40f2-be8c-219cae0a85dc", ""],
               PGIE_HORROR_CLASSIFICATION_ID: ["88b27fff-e6eb-49a2-8a81-da35ada3be6c", ""],
               PGIE_VIOLENCE_CLASSIFICATION_ID: ["", "ebaff746-b5db-4861-9835-acbc6c725a24"],
               PGIE_NUDE_NET_DETECTION_ID: ["aecf0afe-e13a-45fe-a384-6a06c452268d", "ff90ae06-8528-417d-a9c6-8508169a95b0"]}

LST_NAME_VIOLATE = {"48f6b3b4-edd7-4729-a179-d373947b2797": "Đường Lưỡi Bò",
                    "4bcd8d0b-f8d7-4531-90d5-d8b23ce1a14c": "Chứa Quốc Kỳ",
                    "983e8a87-f511-41cd-83fc-e342cae3d75e": "Chứa Đảng Kỳ",
                    "e8286c99-5523-49fa-ab6f-2e41a1139771": "Chứa Quốc Huy",
                    "ff90ae06-8528-417d-a9c6-8508169a95b0": "Ảnh Khiêu Dâm",
                    "fb46026e-0f32-459f-840e-b61bd61fcf9c": "Ảnh Hentai",
                    "32834c32-fe64-461b-b925-da8443836b87": "Ảnh Nghệ Thuật",
                    "ebaff746-b5db-4861-9835-acbc6c725a24": "Ảnh Bạo Lực",
                    "88b27fff-e6eb-49a2-8a81-da35ada3be6c": "Ảnh Kinh Dị",
                    "03e1e976-b9cc-40f2-be8c-219cae0a85dc": "Ảnh Máu Me",
                    "1b15d3d1-d111-4cdb-8c07-bcfa90f5ba78": "Ảnh Nhạy Cảm",
                    "aecf0afe-e13a-45fe-a384-6a06c452268d": "Ảnh Nhạy Cảm (có che)",
                    "ff90ae06-8528-417d-a9c6-8508169a95b0": "Ảnh Nhạy Cảm (không che)"
                    }

URL_SEAWEED = ""


def infer_gore_classification(frame, conf_threshold, meta):
    violate = False
    number_before = len(meta["class"])
    classifications = gore_model.inference(image=frame, id_except=[1])
    if len(classifications) > 0:
        # meta["bbox"].append(x["box"] for x in detections if x["score"]>=conf_threshold)
        meta_update["violate_ids"].append(ID_VIOLATES[PGIE_GORE_CLASSIFICATION_ID][0])
        meta["score"].append(x["score"] for x in classifications if x["score"]>=conf_threshold)
        meta["class"].append(x["class"] for x in classifications if x["score"]>=conf_threshold)
        meta["frame"].append(frame_number for x in classifications if x["score"]>=conf_threshold)
        assert len(meta["score"]) == len(meta["class"]) == len(meta["frame"])
        number_after = len(meta["class"])
        if number_after > number_before:
            violate = True  
    return meta, violate

def infer_horror_classification(frame, conf_threshold, meta):
    violate = False
    number_before = len(meta["class"])
    classifications = horror_model.inference(image=frame, id_except=[1])
    if len(classifications) > 0:
        # meta["bbox"].append(x["box"] for x in detections if x["score"]>=conf_threshold)
        meta["score"].append(x["score"] for x in classifications if x["score"]>=conf_threshold)
        meta["class"].append(x["class"] for x in classifications if x["score"]>=conf_threshold)
        meta["frame"].append(frame_number for x in classifications if x["score"]>=conf_threshold)
        assert len(meta["score"]) == len(meta["class"]) == len(meta["frame"])
        number_after = len(meta["class"])
        if number_after > number_before:
            violate = True  
    return meta, violate

def infer_nudenet_detection(frame, conf_threshold, meta):
    violate = False
    number_before = len(meta["bbox"])
    detections = nudenet_model.detect(frame)
    if len(detections) > 0:
        meta["bbox"].append(x["box"] for x in detections if x["score"]>=conf_threshold)
        meta["score"].append(x["score"] for x in detections if x["score"]>=conf_threshold)
        meta["class"].append(x["class"] for x in detections if x["score"]>=conf_threshold)
        meta["frame"].append(frame_number for x in detections if x["score"]>=conf_threshold)
        assert len(meta["bbox"]) == len(meta["score"]) == len(meta["class"]) == len(meta["frame"])
        number_after = len(meta["bbox"])
        if number_after > number_before:
            violate = True  
    return meta, violate

def infer_dashline_detection(frame, conf_threshold, meta):
    violate = False
    number_before = len(meta["bbox"])
    detections = dashline_model.inference(frame, conf_threshold=conf_threshold)
    if len(detections) > 0:
        meta["bbox"].append(x["box"] for x in detections if x["score"]>=conf_threshold)
        meta["score"].append(x["score"] for x in detections if x["score"]>=conf_threshold)
        meta["class"].append(x["class"] for x in detections if x["score"]>=conf_threshold)
        meta["frame"].append(frame_number for x in detections if x["score"]>=conf_threshold)
        assert len(meta["bbox"]) == len(meta["score"]) == len(meta["class"]) == len(meta["frame"])
        number_after = len(meta["bbox"])
        if number_after > number_before:
            violate = True  
    return meta, violate

def infer_flag_detection(frame, conf_threshold, meta):
    violate = False
    number_before = len(meta["bbox"])
    detections = flag_model.infer(image=frame, conf_thres=conf_threshold)
    if len(detections) > 0:
        meta["bbox"].append(x["box"] for x in detections if x["score"]>=conf_threshold)
        meta["score"].append(x["score"] for x in detections if x["score"]>=conf_threshold)
        meta["class"].append(x["class"] for x in detections if x["score"]>=conf_threshold)
        meta["frame"].append(frame_number for x in detections if x["score"]>=conf_threshold)
        assert len(meta["bbox"]) == len(meta["score"]) == len(meta["class"]) == len(meta["frame"])
        number_after = len(meta["bbox"])
        if number_after > number_before:
            violate = True  
    return meta, violate

def read_video(path_video, id_video, skipframe=5):
    global frame_number
    global meta_update
    meta_update                     = dict()
    meta_update["violate_ids"]      = []
    meta_update["violate_names"]    = []
    meta_update["url_org"]          = ""
    keys                            = ["bbox", "score", "class", "frame"]
    
    print("Starting frames...")
    
    meta_flag = dict()
    meta_dashline = dict()
    meta_nudenet = dict()
    meta_gore = dict()
    meta_horror = dict()
    for key in keys:
        meta_flag[key] = []
        meta_dashline[key] = []
        meta_nudenet[key] = []
        meta_gore[key] = []
        meta_horror[key] = []
    
    
    cap     = cv2.VideoCapture(path_video)
    fps     = cap.get(cv2.CAP_PROP_FPS) + 1
    frame_number=1
    while True:
        grabbed, frame = cap.read()
        print("[INFO] Inferencing {}/{} frames.".format(frame_number, fps))
        url_frame_org = os.path.join(URL_SEAWEED, f"hawkice_demo/{id_video}/{frame_number}_org.jpg")
        urL_frame_upload = upload_url_from_frame_image(image=deepcopy(frame), url_path_img=url_frame_org)
        if frame_number%skipframe==0:
            meta_flag, violate_flag = infer_flag_detection(frame=frame, conf_threshold=0.5, meta=meta_flag)
            meta_dashline, violate_dashline = infer_dashline_detection(frame=frame, conf_threshold=0.8, meta=meta_dashline)
            meta_nudenet, violate_nudenet = infer_nudenet_detection(frame=frame, conf_threshold=0.8, meta=meta_dashline)
            meta_gore, violate_gore = infer_gore_classification(frame=frame, conf_threshold=0.5, meta=meta_gore)
            meta_horror, violate_horror = infer_horror_classification(frame=frame, conf_threshold=0.5, meta=meta_gore)
            
            if frame_number%20==0 and not violate_flag:
                
                confidence_avg, urL_draw_upload = upload_data_each_frame(url_seaweed=URL_SEAWEED, 
                                                        meta_datas=meta_detections[key], 
                                                        id_video=id_video, 
                                                        frame_number=frame_number, 
                                                        id_model=key, 
                                                        class_name=LST_NAME_VIOLATE[key], 
                                                        frame=frame_draw)
                url_image_violates = {
                                        "url_org": urL_frame_upload.replace("http://10.9.3.50:8888", "https://dc2.file.icomm.vn"),
                                        "url_cut": urL_draw_upload.replace("http://10.9.3.50:8888", "https://dc2.file.icomm.vn"),
                                        "score": confidence_avg
                                    }
                
            
        print("[INFO] Next frame...")
        frame_number+=1

    video.release()

if __name__ == "__main__":
    read_video(path_video="Untitled.mp4")