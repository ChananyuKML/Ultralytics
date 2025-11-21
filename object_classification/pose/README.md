# Pose Estimation
| Model Type           | Pre-trained Weight | Task Supported  |
| :------------------  | :----------------- | :-------------- |
| YOLO12n-pose         | yolo12n.pt         | Pose Extimation |
| YOLO12s-pose         | yolo12s.pt         | Pose Extimation |
| YOLO12m-pose         | yolo12m.pt         | Pose Extimation |
| YOLO12l-pose         | yolo12l.pt         | Pose Extimation |
| YOLO12x-pose         | yolo12x.pt         | Pose Extimation |

## Dataset Structure
```bash
example_dataset/
├── example.yaml
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   ├── img3.jpg
│   │   └── img4.jpg
│   └── val/
│       ├── img5.jpg
│       ├── img6.jpg
│       ├── img7.jpg
│       └── img8.jpg
└── labels/
    ├── train/
    │   ├── img1.txt
    │   ├── img2.txt
    │   ├── img3.txt
    │   └── img4.txt
    └── val/
        ├── img5.txt
        ├── img6.txt
        ├── img7.txt
        └── img8.txt
```
## example.yaml
```bash
path: example_dataset # dataset root dir
train: images/train # train images (relative to 'path') 4 images
val: images/val # val images (relative to 'path') 4 images
test: # test images (optional)

# Keypoints
kpt_shape: [17, 3] # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# Classes
names:
  0: person

# Keypoint names per class
kpt_names:
  0:
    - nose
    - left_eye
    - right_eye
    - left_ear
    - right_ear
    - left_shoulder
    - right_shoulder
    - left_elbow
    - right_elbow
    - left_wrist
    - right_wrist
    - left_hip
    - right_hip
    - left_knee
    - right_knee
    - left_ankle
    - right_ankle
```

## Label.txt Files  
Each img.txt file contains information of bounding boxes in corresponding image and class of the object inside bounding box.

```bash
<class> <x1> <y1> <x2> <y2> .... <x17> <y17>
0 0.671279 0.617945 0.645759 0.726859 .... 0.000000 0.000000
```

## Train
```bash
python train.py --pt <pre-trained-weight> --dataset <path-to-dataset-folder> --epochs <number-of-epochs>
```

## Inference
```bash
python inference.py <path-to-image> --pt <path-to-model-file>
```