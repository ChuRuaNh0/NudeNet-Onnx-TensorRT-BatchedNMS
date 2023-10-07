from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data='yolov8.yaml',
   imgsz=640,
   epochs=100,
   batch=16,
   name='yolov8n_v8_50e'
)