from ultralytics import YOLO
import os

dataset_path = r"C:\Users\david\Desktop"
model_path = r"C:\Users\david\projects\ultralytics"


dataset = "merged_yolo"
yolo_model = "yolov8n-seg"

# Model config
epochs = 300
patience = 10
batch = 8 # -1 auto
imgsz = (1024,768)
save = True
name = yolo_model + "_" + str(epochs) + "_" + dataset
exist_ok = True
pretrained = True
single_cls = False
device = "cuda"
overlap_mask = False
mask_ratio = 2
plots = True


# Augmentation configs
hsv_h = 0.015
hsv_s = 0.5
hsv_v = 0.3
degrees = 0.0
translate = 0.1
scale = 0.2
shear = 1.0
flipud = 0.5
fliplr = 0.5
mosaic = 0.0
mixup = 0.0
copy_paste = 0.3


model = YOLO(yolo_model + ".yaml").load(os.path.join(model_path, yolo_model + ".pt"))

model.train(data=os.path.join(dataset_path, dataset, "data_local.yaml"),
            epochs=epochs,
            patience=patience,
            imgsz=imgsz,
            device=device,
            name=name,
            exist_ok=exist_ok,
            mosaic=mosaic,
            flipud=flipud,
            fliplr=fliplr,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            pretrained=pretrained,
            single_cls=single_cls,
            overlap_mask=overlap_mask,
            mask_ratio=mask_ratio,
            mixup=mixup,
            copy_paste=copy_paste)