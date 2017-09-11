#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/reinforcement_scalefac_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReinforcementScalefacBackward(const int nthreads,
    const Dtype* in_diff, const Dtype* R, const Dtype* b, const int C, const Dtype* curr, const Dtype lambda, 
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {

    if(index>=nthreads) return;

    int item = index/C;
    int ic   = index%C;

    Dtype curr_weight=1.;

    if(curr) curr_weight=curr[item];
    out_diff[item*C+ic]=lambda*curr_weight*(R[item]-b[item])*in_diff[item*C+ic];

  }

}

template <typename Dtype>
void ReinforcementScalefacLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  caffe_copy(bottom[0]->count(),
             bottom[0]->gpu_data(),
             top[0]->mutable_gpu_data());
}


template <typename Dtype>
void ReinforcementScalefacLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


  Dtype* out_diff=bottom[0]->mutable_gpu_diff();
  const Dtype* in_diff=top[0]->gpu_diff();
  const Dtype* b=bottom[1]->gpu_data();
  const Dtype* R=top[1]->gpu_diff();
  const Dtype* curr=NULL;
  if(top.size()>2) curr=top[2]->gpu_diff();
  int C=bottom[0]->count(1);

  ReinforcementScalefacBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
      bottom[0]->count(), in_diff, R, b, C, curr, lambda_, out_diff);

  CUDA_POST_KERNEL_CHECK;

}

INSTANTIATE_LAYER_GPU_FUNCS(ReinforcementScalefacLayer);


}  // namespace caffe
