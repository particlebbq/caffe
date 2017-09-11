#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/bernoulli_sample_layer.hpp"

namespace caffe {



template <typename Dtype>
__global__ void BernoulliSampleForward(const int nthreads, const Dtype* p, 
      const int D, Dtype* sample){

  CUDA_KERNEL_LOOP(index, nthreads) {

    if(index>=nthreads) return;

    int item = index/D;
    int idim   = index%D;

    if(sample[item*D+idim]>p[item*D+idim]) {
      sample[item*D+idim]=0;
    } else {
      sample[item*D+idim]=1;
    }
  }

}


template <typename Dtype>
void BernoulliSampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


  const Dtype* p=bottom[0]->gpu_data();
  Dtype* sample=top[0]->mutable_gpu_data();
  int N=bottom[0]->shape(0);
  int D=bottom[0]->count(1);

  caffe_gpu_rng_uniform<Dtype>(bottom[0]->count(),0.,1.,sample);

    BernoulliSampleForward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->count(), p, D, sample);


  CUDA_POST_KERNEL_CHECK;


}


template <typename Dtype>
void BernoulliSampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(BernoulliSampleLayer);


}  // namespace caffe
