# RT-Detr
| Model Type          | Pre-trained Weight | Task Supported   |
| :------------------ | :----------------- | :--------------- |
| RT-DETR Large       | rtdetr-l.pt        | Object Detection |
| RT-DETR Extra-Large | rtdetr-x.pt        | Object Detection |
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

Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
```

## Label.txt Files  
Each img.txt file contains information of bounding boxes in corresponding image and class of the object inside bounding box.
```bash
<class> <center-x> <center-y> <width> <height>
58 0.519219 0.451121 0.39825 0.75729
```

## Train
```bash
python train.py --pt <pre-trained-weight> --dataset <path-to-dataset-folder> --epochs <number-of-epochs>
```

## Inference
```bash
python inference.py <path-to-image> --pt <path-to-model-file>
```