# Introduction
This project contains work-in-progress (WIP) code for experimenting with Computer Vision using OpenCV in Python. Little 
to no attempt will be made to keep this project up-to-date and as a result, the dependencies will probably eventually 
break and make you cry.

## Target Platform & Dependencies
While experimenting, the target platform is either Apple Silicon (M2 Max) -OR- Ubuntu Server 24.04.2 LTS.
For maximum performance, you will likely need to compile OpenCV for your target platform.

Dependencies are managed via [Anaconda](https://anaconda.org/anaconda/python).

## TL;DR
```bash
# setup your environment
conda env create -f environment.yml

# New hotness
python -m test.motion -i 'path/to/video.mp4'

# Old & busted
python -m deprecated.quick -i 'path/to/video.mp'
```

## FAQ
#### I want to read your code, where do I begin?
> Start with [./masks/motion_mog2.py](./masks/motion_mog2.py)
> 
> This class detects motion by creating masks with [MOG2 Background Subtraction](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html)
> and then applying an [Adaptive Image Threshold](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) to reduce noise.
> To further increase quality, it also applies down-sampling, gaussian blur and other techniques such as erosion and dilation
> to improve the mask's image quality. Once a mask has been created, further processing can detect contours and create 
> bounding boxes etc.
