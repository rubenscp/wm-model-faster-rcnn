# White Mold Faster R-CNN Model

## Institute of Computing (IC) at University of Campinas (Unicamp)

## Postgraduate Program in Computer Science

### Team

* Rubens de Castro Pereira - student at IC-Unicamp
* Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
* Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
* Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans

### Main purpose

This Python project aims to train and inference the Faster R-CNN model in the image dataset of white mold disease and its stages.

This implementation is based on this notebook:
<https://www.kaggle.com/code/yerramvarun/fine-tuning-faster-rcnn-using-pytorch>

## Installing Python Virtual Environment

```
module load python/3.10.10-gcc-9.4.0
```
```
pip install --user virtualenv
```
```
virtualenv -p python3.10 venv-wm-model-faster-rcnn
```
```
source venv-wm-model-faster-rcnn/bin/activate
```
```
pip install -r requirements.txt
```

## Running Python Application

```
access specific folder 'wm-model-faster-rcnn'
```
```
python my-python-modules/manage_faster_rcnn_train.py
```

## Submitting Python Application at LoveLace environment

Version of CUDA module to load:
- module load cuda/11.5.0-intel-2022.0.1

```
qsub wm-model-ssd.script
```
```
qstat -u rubenscp
```
```
qstat -q umagpu
```

The results of job execution can be visualizedat some files as:

* errors
* output
