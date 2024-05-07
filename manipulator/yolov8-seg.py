#%%
from ultralytics import YOLO
import torchvision.transforms as T
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
#%%
datasets = r"D:\datasets"
dataset = "50img_yolo"
yolo_model = "yolov8n-seg" # yolov8l-seg.pt yolov8m-seg.pt yolov8n-seg.pt yolov8s-seg.pt yolov8x-seg.pt
epochs = 30
name = yolo_model + "_" + str(epochs) + "_" + dataset
#%%

#%%
model = YOLO(yolo_model + ".yaml").load(yolo_model + ".pt")  
# see https://docs.ultralytics.com/modes/train/ for more details
'''yolo detect train \
data=10Class.yaml \
model=yolov8s.pt \
epochs=300 \
imgsz=1280 \
rect=True \
device=0 \
batch=-1 \
plots=True \
hsv_h=0.015 \
hsv_s=0.5 \
hsv_v=0.3 \
degrees=0.0 \
translate=0.1 \
scale=0.2 \
shear=1.0 \
flipud=0.5 \
fliplr=0.5 '''

'''task=detect, mode=train, model=yolov8n.pt, data=coco128.yaml, epochs=3, patience=50, batch=16, imgsz=640, save=True, cache=False, device=, workers=8, project=None, name=None, exist_ok=False, pretrained=False, optimizer=SGD, verbose=False, seed=0, deterministic=True, single_cls=False, image_weights=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, show=False, save_txt=False, save_conf=False, save_crop=False, hide_labels=False, hide_conf=False, vid_stride=1, line_thickness=3, visualize=False, augment=False, agnostic_nms=False, retina_masks=False, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=17, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, fl_gamma=0.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, cfg=None, hydra={'output_subdir': None, 'run': {'dir': '.'}}, v5loader=False, save_dir=runs/detect/train'''
model.train(data=os.path.join(datasets, dataset, "data.yaml"),
            epochs=epochs,
#            patience=3,
            imgsz=(1024,768),
            device="0",
            name=name,
            mosaic=0.0,
            scale=0.2,
            flipud=1.0,
            fliplr=1.0,
            hsv_h=0.2,
            hsv_s=0.2,
            hsv_v=0.2,
            shear=0.2,
            pretrained=True,
            degrees=1.0)
metrics = model.val()
#%%
success = model.export(format="onnx")  # export the model to ONNX format

#%%
# predict on folder
predict_imgs = r"D:\datasets\testing\images"
model = YOLO("D:/models/ultralytics/runs/segment/%s/weights/best.pt" % name)
results = model.predict(source=predict_imgs, show=False, save=True)
#%%
# predict on an image
im1 = Image.open(r"D:\datasets\testing\images\Marvell_tran_001.tif")
results = model(im1)  # predict on an image
result = results[0]
T.ToPILImage()(result.masks.masks).show()
#%%