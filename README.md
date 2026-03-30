## A model-driven deep neural network for simultaneous low-light image enhancement and deblurring

### 🔔 **The pretrained weights and qualitative results will be updated after the paper is accepted.**

---

### Dependencies and Installation

- Pytorch >= 1.13.1
- CUDA >= 11.6
- Other required packages in `requirements.txt`
```
# git clone this repository
git clone https://github.com/MingyuLiu1/LIEDNet.git
cd LIEDNet

# create new anaconda env
conda create -n liednet python=3.10 -y
conda activate liednet

# install python dependencies
pip3 install -r requirements.txt
python basicsr/setup.py develop

cd basicsr/kernels/selective_scan
python install .
```

### Train the Model
- Specify the path to training and evaluation data in the corresponding option file.

Training:
```
# LOL-Blur
python train_lolblur.py --opt options/train_LOLBLur.yml
```

```
# LOL-v1
python train_llie.py --opt options/train_LOLv1.yml 
```

```
# LOL-v2-syn
python train_llie.py --opt options/train_LOLv2_synthetic.yaml
```

```
# FiveK
python train_llie.py --opt options/train_fivek.yaml
```

### Quick Inference

Inference (save images):
```
# test on LOL-Blur
python inference_lol_blur.py --test_path $INPUT PATH$ --result_path $SAVE PATH$ --ckpt $CHECKPOINT PATH$

# test on LLIE datasets (LOL-V1, LOL-V2-synth, FiveK)
python inference_llie.py --test_path $INPUT PATH$ --result_path $SAVE PATH$ --ckpt $CHECKPOINT PATH$
```
The results will be saved in `SAVE PATH`.
