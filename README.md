# Face Recognition Project
###

Implementation of the MTCNN face detector and FaceNet. The project uses model from the papers:
 - [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878)
 - [FaceNet: A Unified Embedding for Face Recognition and Clustering](http://arxiv.org/abs/1503.03832)

It is written from scratch, using as a reference the implementation of MTCNN and FaceNet from David Sandberg ([FaceNet](https://github.com/davidsandberg/facenet/)) in Facenet. 

----
## 1. Requirements
It used Python3.6 and Tensorflow 1.14. Include packages:
- tensorflow==1.14
- scipy==1.1.0
- opencv-python
- opencv-python-headless
- scikit-learn==0.22.2
- numpy==1.19.4
- Keras==2.2.5

It can be installed package through pip:

    $ pip install -r requirements.txt

---

## 2. Installation
2.1. Clone this repo and 
move to this directory

    $ git clone https://github.com/nddung105/face_recognition.git
    $ mv face_recognition

2.2. You have to download the Pre-trained models:

- [Pre-trained FaceNet](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz): unzip and put this folder in the directory "**frozen_facenet**".

 -  [MTCNN Weights](https://drive.google.com/file/d/16fnKUxtcqDVDnszm4YlwgIa5XyrnIbgG/view?usp=sharing):  put it in the directory "**weights**". 
This file includes weight of PNet, RNet and ONet 

2.3. Set the environment variable PYTHONPATH to point to the **'mtcnn'** directory of the cloned repo. 

    $ export PYTHONPATH=$PWD/mtcnn

--- 
## 3. Usage
Use it to open the camera and recognize faces

    $ python webcam_infer.py