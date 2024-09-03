from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

#results = model(source=1, show= True, conf =0.4, save=True)
#results = model.predict(source = 0, show =True)
results = model.predict( source = '0', show =True)
#'test.mp4'