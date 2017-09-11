import cv2
import numpy as np
import re
import caffe
import argparse
import random


parser = argparse.ArgumentParser(description='generate some digit samples to monitor GAN progress')
parser.add_argument("gpu", metavar='N', type=int, nargs=1, help='which gpu to run on')

args = parser.parse_args()
gpu=args.gpu[0]



model_file="./examples/mnist/train_val_GAN.prototxt"
pretrained_file="examples/mnist/caffenet_GAN_iter_1145000.caffemodel"

caffe.set_mode_gpu()
caffe.set_device(gpu)

net=caffe.Net(model_file,pretrained_file,caffe.TRAIN) 
batch_size=2

i_img=0
composite=np.zeros((280,280,1),dtype=np.uint8)
for ibatch in range(3):
  net.forward()
  batch_out=net.blobs["generated"].data

  for item in range(batch_out.shape[0]):
    if i_img>=100:
      break

    pix=np.zeros((batch_out.shape[2],batch_out.shape[3],batch_out.shape[1]),dtype=np.uint8)
    pix[:,:,0]=(batch_out[item,0,:,:]*255.999).astype(np.uint8)

    ix=i_img/10
    iy=i_img%10
    composite[28*ix:28*(ix+1),28*iy:28*(iy+1),0]=(batch_out[item,0,:,:]*255.999).astype(np.uint8)

    pix2=np.zeros((batch_out.shape[2],batch_out.shape[3],batch_out.shape[1]),dtype=np.int)
    pix2[:,:,0]=(batch_out[batch_out.shape[0]-item-1,0,:,:]*255.999).astype(np.uint8)

    i_img+=1

cv2.imwrite("composite.png",composite)
