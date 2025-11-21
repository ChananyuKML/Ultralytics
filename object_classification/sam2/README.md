# SAM2
| Model Type          | Pre-trained Weight | Task Supported        |
| :------------------ | :----------------- | :-------------------- |
| SAM 2.1 tiny        | sam2.1_t.pt        | Instance Segmentation |
| SAM 2.1 small       | sam2.1_s.pt        | Instance Segmentation |
| SAM 2.1 base        | sam2.1_b.pt        | Instance Segmentation |
| SAM 2.1 large       | sam2.1_l.pt        | Instance Segmentation |

## Inference
```bash
python inference.py <path-to-image> --pt <path-to-model-file>
```

## Train
```bash
python train.py --pt <pre-trained-weight> --dataset <path-to-dataset-folder> --epochs <number-of-epochs>
```

## Inference
```bash
python inference.py <path-to-image> --pt <path-to-model-file>
```