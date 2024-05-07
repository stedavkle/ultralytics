#%%
from ultralytics import YOLO
import os
modelname = "yolov8-multitask"
model = YOLO(modelname + ".yaml")

model.load("C:/Users/david/projects/ultralytics/yolov8n.pt")
model.train(data="C:/Users/david/projects/datasets/merged8_yolo_multitask/merged8-multitask_flip.yaml", patience=2, epochs=1)


#%%
from ultralytics import YOLO
import os
modelname = "yolov8-multitask"
model = YOLO(modelname + ".yaml")

model.load("C:/Users/david/projects/ultralytics/yolov8n.pt")
model.train(data="C:/Users/david/projects/datasets/coco8-multitask/coco8-multitask.yaml", patience=2, epochs=1)

#%%
from ultralytics import YOLO
import os
modelname = "yolov8n-seg"
model = YOLO(modelname + ".yaml")

model.load("C:/Users/david/projects/ultralytics/yolov8n.pt")
model.train(data="coco8-seg.yaml", patience=2, plots=True)

#%%
from ultralytics import YOLO
import os
modelname = "yolov8n-pose"
model = YOLO(modelname + ".yaml")

model.load("C:/Users/david/projects/ultralytics/yolov8n.pt")
model.train(data="coco8-pose.yaml", patience=2, plots=True, epochs=1)
#%%
from ultralytics import YOLO
model = YOLO(r'C:\Users\david\projects\ultralytics\yolov8n-multi-kn-mc.pt')
dets = model(r'C:\Users\david\Downloads\val', save=True, max_det=8, conf=0.6, augment=True)

model.train(data=r'C:\Users\david\projects\datasets\merged8_yolo_multitask\merged8-multitask.yaml', patience=2, epochs=1)
dets = model(r'C:\Users\david\Downloads\val', save=True, max_det=8, conf=0.6, augment=True)
model.val(data=r'C:\Users\david\projects\datasets\merged8_yolo_multitask\merged8-multitask.yaml')

#%%
from ultralytics.data.utils import verify_image_label
# verify segmentation image/label pair
im_file = r'C:\Users\david\projects\datasets\coco8-multitask\images\train\000000001490.jpg'
lb_file = r'C:\Users\david\projects\datasets\coco8-multitask\labels-seg\train\000000001490.txt'
prefix = ''
keypoint = False
num_cls = 80
nkpt = 0
ndim = 0
seg = verify_image_label((im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim, None))
print(seg)
#%%
# verify keypoint image/label pair
im_file = r'C:\Users\david\projects\datasets\coco8-multitask\images\train\000000001490.jpg'
lb_file = r'C:\Users\david\projects\datasets\coco8-multitask\labels-pose\train\000000001490.txt'
prefix = ''
keypoint = True
num_cls = 1
nkpt = 17
ndim = 3
kpt = verify_image_label((im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim, None))
#print(kpt)
#%%
# verify multi-task image/label pair
im_file = r'C:\Users\david\projects\datasets\coco8-multitask\images\train\000000001490.jpg'
lb_file = r'C:\Users\david\projects\datasets\coco8-multitask\labels\train\000000001490.txt'
prefix = ''
keypoint = True
num_cls = 80
nkpt = 17
ndim = 3
kpt_names = {0: 'person'}
segpose = verify_image_label((im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim, kpt_names))
print(segpose)

#%%
from ultralytics.data.dataset import YOLODataset
import yaml
# load r'C:\Users\david\projects\datasets\coco8-multitask\coco8-multitask.yaml'
data = yaml.safe_load(open(r'C:\Users\david\projects\datasets\merged8_yolo_multitask\merged8-multitask.yaml', 'r', encoding='utf-8'))
print(data['kpt_names'])
dataset = YOLODataset(data=data, use_segments=True, use_keypoints=True, img_path=r'C:\Users\david\projects\datasets\merged8_yolo_multitask\images\train')
from ultralytics.data.dataset import YOLODataset
import yaml
pose_data = yaml.safe_load(open(r'C:\Users\david\projects\datasets\coco8-pose\coco8-pose.yaml', 'r', encoding='utf-8'))
pose_dataset = YOLODataset(data=pose_data, use_segments=False, use_keypoints=True, img_path=r'C:\Users\david\projects\datasets\coco8-pose\images\train')
# MERGE SEGMENTS AND POSE
# %%
import os
import glob
from ultralytics.utils.ops import segments2boxes
import numpy as np

def match_pose_to_segment(seg_line, pose_lines):
    seg_parts = [x.split() for x in seg_line.strip().splitlines() if len(x)]
    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in seg_parts]  # (cls, xy1...)
    seg_bbox = segments2boxes(segments)[0]

    best_match = None
    min_bbox_diff = float('inf')

    lb = [x.split() for x in pose_lines if len(x)]

    for i, pose_bbox in enumerate([np.array(x[1:5], dtype=np.float32) for x in lb]):
        bbox_diff = sum(abs(seg_bbox[i] - pose_bbox[i]) for i in range(4))
        if bbox_diff < min_bbox_diff:
            min_bbox_diff = bbox_diff
            best_match = pose_lines[i]

    return best_match

def merge_annotations(seg_path, pose_path, output_base_path):
    for subdir, _, _ in os.walk(seg_path):
        relative_path = os.path.relpath(subdir, seg_path)
        pose_subdir = os.path.join(pose_path, relative_path)
        output_subdir = os.path.join(output_base_path, relative_path)

        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        seg_files = glob.glob(os.path.join(subdir, '*.txt'))

        for seg_file in seg_files:
            pose_file = os.path.join(pose_subdir, os.path.basename(seg_file))
            output_file = os.path.join(output_subdir, os.path.basename(seg_file))

            if os.path.exists(pose_file):
                with open(seg_file, 'r') as seg, open(pose_file, 'r') as pose, open(output_file, 'w') as out:
                    seg_lines = seg.readlines()
                    pose_lines = pose.readlines()

                    for seg_line in seg_lines:
                        seg_class_index = seg_line.strip().split()[0]
                        if seg_class_index == "0":  # Process only if class index is 0
                            best_match = match_pose_to_segment(seg_line, pose_lines)
                            if best_match:
                                pose_parts = best_match.strip().split()
                                seg_parts = seg_line.strip().split()
                                merged_line = pose_parts[0] + ' ' + ' '.join(pose_parts[5:]) + ' ' + ' '.join(seg_parts[1:]) + '\n'
                                out.write(merged_line)
                        else:
                            # Write segmentation line without pose points
                            out.write(seg_line)

# Example usage
seg_annotations_path = r'C:\Users\david\projects\datasets\coco8-multitask\labels-seg\val'
pose_annotations_path = r'C:\Users\david\projects\datasets\coco8-multitask\labels-pose\val'
output_annotations_path = r'C:\Users\david\projects\datasets\coco8-multitask\labels\val'

merge_annotations(seg_annotations_path, pose_annotations_path, output_annotations_path)
# %%
# copy images to train/val folders
import os
import shutil
for subdir, _, files in os.walk(r'G:\My Drive\merged_yolo_multitask\images'):
    for file in files:
        if file.endswith('.tif'):
            src = os.path.join(subdir, file)
            if os.path.exists(os.path.join(r'G:\My Drive\merged_yolo_multitask\labels\train', file.replace('.tif', '.txt'))):
                dst = os.path.join(r'G:\My Drive\merged_yolo_multitask\images\train', file)
            else:
                dst = os.path.join(r'G:\My Drive\merged_yolo_multitask\images\val', file)
            shutil.move(src, dst)
# %%
