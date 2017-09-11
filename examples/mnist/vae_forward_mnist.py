import json
import cv2
import numpy as np
import re
import caffe
import argparse
from ROOT import TFile, TCanvas, TH1D
import random


parser = argparse.ArgumentParser(description='run trained VAE to produce samples')
parser.add_argument("gpu", metavar='N', type=int, nargs=1, help='which gpu')

args = parser.parse_args()
gpu=args.gpu[0]



model_file="./examples/mnist/train_val_VAE.prototxt"
pretrained_file="examples/mnist/caffenet_VAE_iter_1200000.caffemodel"

caffe.set_mode_gpu()
caffe.set_device(gpu)  #1)

net=caffe.Net(model_file,weights=pretrained_file,phase=caffe.TRAIN)  #keep it in "training" mode so that we have dropout for data augmentation
batch_size=2

for iaxis in range(10):
  for jaxis in range(iaxis,10):
    
    composite=np.zeros((280,280,1),dtype=np.uint8)
    
    net.forward()
    sample=net.blobs["decoder_sample"].data

    for isamp in range(10):
      for jsamp in range(10):
        img=(sample[isamp*10+jsamp].reshape((28,28))*255).astype(np.uint8)
        composite[isamp*28:(isamp+1)*28,jsamp*28:(jsamp+1)*28,0]=img[:,:]

    cv2.imwrite("debug/composite_vae_"+str(iaxis)+"_"+str(jaxis)+".png",composite)

