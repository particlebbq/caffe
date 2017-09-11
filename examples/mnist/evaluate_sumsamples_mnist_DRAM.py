import lmdb
import caffe

import cv2
import numpy as np
import scipy 
import os
import re
import pickle
from math import *

model_file="./examples/mnist/train_val_DRAM.prototxt"
pretrained_file_list=["examples/mnist/caffenet_DRAM_iter_300000.caffemodel"]

pretrained_file=pretrained_file_list[0]

caffe.set_mode_gpu()
caffe.set_device(1)

net=caffe.Net(model_file,pretrained_file,caffe.TEST)

total_blob_count=0
for blobname in net.blobs:
  total_blob_count=total_blob_count+net.blobs[blobname].count

frac=0.
frac_avg=0.
frac_logavg=0.
n_batches=1
n_inputpatterns=10000

for pretrained_file in [pretrained_file_list[0]]:
  for ipat in range(n_inputpatterns):
    if ipat%100==0:
      print("working on input "+str(ipat)+" of "+str(n_inputpatterns))

    prob_avg=[0 for iclass in range(10)]  
    prob_log_avg=[0 for iclass in range(10)]  

    for ibatch in range(n_batches):
      net.forward()
      predexp=np.exp(net.blobs["predict_output"].data)
      sumexp=np.broadcast_to(np.sum(predexp,axis=1).reshape((512,1,8)),(512,10,8))
      probs=np.divide(predexp,sumexp)
  
      label=net.blobs["label"].data
  
      prob_avg_0=[0 for iclass in range(probs.shape[1])]
      prob_avg_1=[0 for iclass in range(probs.shape[1])]
  
      prob_log_avg_0=[0 for iclass in range(probs.shape[1])]
      prob_log_avg_1=[0 for iclass in range(probs.shape[1])]
  
      idx0=int(label[0,0])
      idx1=int(label[0,1])
  
      for item in range(probs.shape[0]):
        prob0=probs[item,idx0,3]
        prob1=probs[item,idx1,7]
        argmax_0=0
        argmax_1=0
        for iclass in range(probs.shape[1]):
          prob_avg_0[iclass]+=probs[item,iclass,3]
          prob_avg_1[iclass]+=probs[item,iclass,7]
          prob_log_avg_0[iclass]+=np.log(probs[item,iclass,3])
          prob_log_avg_1[iclass]+=np.log(probs[item,iclass,7])
          if probs[item,iclass,3]>probs[item,argmax_0,3]:
            argmax_0=iclass
          if probs[item,iclass,7]>probs[item,argmax_1,7]:
            argmax_1=iclass
        if argmax_0==idx0:
          frac+=0.5
        if argmax_1==idx1:
          frac+=0.5
      
      
      prob_avg_0=[p/probs.shape[0] for p in prob_avg_0]
      prob_avg_1=[p/probs.shape[0] for p in prob_avg_1]
      prob_log_avg_0=[np.exp(p/(probs.shape[0])) for p in prob_log_avg_0]
      prob_log_avg_1=[np.exp(p/(probs.shape[0])) for p in prob_log_avg_1]

      argmax_0=0
      argmax_1=0
      for iclass in range(probs.shape[1]):
        if prob_avg_0[iclass]>prob_avg_0[argmax_0]:
          argmax_0=iclass
        if prob_avg_1[iclass]>prob_avg_1[argmax_1]:
          argmax_1=iclass
      if argmax_0==idx0 and argmax_1==idx1:
        frac_avg+=1
  
  
      argmax_0=0
      argmax_1=0
      for iclass in range(probs.shape[1]):
        if prob_log_avg_0[iclass]>prob_log_avg_0[argmax_0]:
          argmax_0=iclass
        if prob_log_avg_1[iclass]>prob_log_avg_1[argmax_1]:
          argmax_1=iclass
      if argmax_0==idx0 and argmax_1==idx1:
        frac_logavg+=1
 

  frac/=n_batches*probs.shape[0]*n_inputpatterns
  frac_avg/=n_inputpatterns
  frac_logavg/=n_inputpatterns

  print("frac="+str(frac)+", frac_avg="+str(frac_avg)+", frac_logavg="+str(frac_logavg))
 
print("\n\n\n\n")
