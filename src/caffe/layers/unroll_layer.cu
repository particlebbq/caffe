#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/unroll_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template <typename Dtype>
void UnrollLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  for(int itime=0;itime<num_timesteps_;itime++){
    for(int isubnet=0;isubnet<subnet_blobs_[itime].size();isubnet++){
      for(int iblob=0;iblob<subnet_blobs_[itime][isubnet].size();iblob++) {
        caffe_gpu_set<Dtype>(subnet_blobs_[itime][isubnet][iblob]->count(),
            Dtype(0.),subnet_blobs_[itime][isubnet][iblob]->mutable_gpu_diff());
      }
      subnet_layer_->subnets()[isubnet]->reset_blobs(
          subnet_blobs_[itime][isubnet]);
    }
    subnet_layer_->Forward(subnet_bottom_vec_[itime],subnet_top_vec_[itime]);
    for(int i=0;i<N_recur_;i++) {
      split_layer_[itime][i]->Forward(split_bottom_vec_[itime][i],
          split_top_vec_[itime][i]);
    }
    for(int i=0;i<N_output_+N_recur_;i++){
      reshape_layer_[itime][i]->Forward(reshape_bottom_vec_[itime][i],
          reshape_top_vec_[itime][i]);
    }
  }
  for(int i=0;i<N_output_+N_recur_;i++){
    concat_layer_[i]->Forward(concat_bottom_vec_[i],concat_top_vec_[i]);
  }

}

INSTANTIATE_LAYER_GPU_FORWARD(UnrollLayer);


}  // namespace caffe
