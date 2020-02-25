# mmdetection_cpu_inference (only supporting CascadeRCNN currently)

[mmdetection](https://github.com/open-mmlab/mmdetection) is a powerful codebase of object detection and instance segmentation for both academic research and real life applications. In order to train powerful models you need decent GPUs and the code itself requires GPU to start compiling.\
However, there might be only CPUs available when you put the GPU-trained model into production. This repo enables running the models trained with [mmdetection](https://github.com/open-mmlab/mmdetection) on CPU.

## Install
All the operations and moduals have been changed to these in the pre-compiled pytorch. So you just need to install the CPU-version pytorch:
```python
conda install pytorch torchvision cpuonly -c pytorch
```

## Run inference
You just need to have the GPU trained model ready.
```python
python inference.py [config_file] [model_checkpoint(.pth)] [input_folder] [output_folder]
```
You only need these several lines:
```python
from mmdet.apis import init_detector

model = init_detector(config_file, model_checkpoint)
bboxes, labels = model.detect(img)
```
