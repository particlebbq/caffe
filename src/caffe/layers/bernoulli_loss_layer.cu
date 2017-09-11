#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/bernoulli_loss_layer.hpp"

namespace caffe {



template <typename Dtype>
__global__ void BernoulliLossForward(const int nthreads,
    const Dtype* x, const Dtype* y, const int D, const int N,
    Dtype* loss_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {

    if(index>=nthreads) return;

    int item = index/D;
    int idim   = index%D;

    loss_data[item*D+idim]+=(x[item*D+idim]*log(fmaxf(0.01,y[item*D+idim]))
                          +(1-x[item*D+idim])*log(fmaxf(0.01,1-y[item*D+idim])))/N;


  }
}


template <typename Dtype>
__global__ void BernoulliLossBackward(const int nthreads,
    const Dtype* x, const Dtype* y, const int D, const int N, const Dtype* loss_wt,
    Dtype* dy) {
  CUDA_KERNEL_LOOP(index, nthreads) {

    if(index>=nthreads) return;

    int item = index/D;
    int idim   = index%D;

    dy[item*D+idim]=-1*(x[item*D+idim]/fmaxf(0.01,y[item*D+idim])
                    -(1-x[item*D+idim])/(fmaxf(0.01,1-y[item*D+idim])))/N;

  }

}


template <typename Dtype>
__global__ void BernoulliClipGrad(const int nthreads,const int D, const Dtype* loss_wt,
    const int idx, const Dtype maxval, const Dtype* in_data,
    const Dtype* breakdown, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {

    if(index>=nthreads) return;

    int item = index/D;

    Dtype scalefac=1.;
    if(breakdown){
      scalefac*=breakdown[item];
    } else {
      scalefac*=loss_wt[0];
    }
    Dtype datmax=fabs(in_data[idx]*scalefac);
    if(datmax>maxval) scalefac*=maxval/fabs(datmax);

    out_data[index]=scalefac*in_data[index];

  }
}




template <typename Dtype>
void BernoulliLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


  const Dtype* y=bottom[0]->gpu_data();
  const Dtype* x=bottom[1]->gpu_data();
  Dtype* loss=top[0]->mutable_cpu_data();
  Dtype* breakdown=NULL;
  if(top.size()>=2){
    breakdown=top[1]->mutable_cpu_data();
  }
  int N=bottom[0]->shape(0);
  int D=bottom[0]->count(1);

  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();


    BernoulliLossForward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->count(), x, y, D, N, loss_data);

  CUDA_POST_KERNEL_CHECK;


  int nthreads=bottom[0]->count();
  caffe_gpu_asum<Dtype>(nthreads, loss_data, loss);
  if(breakdown){
    for(int i=0;i<N;i++){
      caffe_gpu_asum(D,loss_data+i*D,breakdown+i);
    }
  }

  CUDA_POST_KERNEL_CHECK;

}


template <typename Dtype>
void BernoulliLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* y=bottom[0]->gpu_data();
  const Dtype* x=bottom[1]->gpu_data();
  const Dtype* loss_wt=top[0]->gpu_diff();

  const Dtype* breakdown=NULL;
  if(top.size()>=2) {
    breakdown=top[1]->gpu_diff();
  }

  Dtype* dy=bottom[0]->mutable_gpu_diff();


  int N=bottom[0]->shape(0);
  int D=bottom[0]->shape(1);

    BernoulliLossBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->count(), bottom[1]->gpu_data(), bottom[0]->gpu_data(), D, N, loss_wt, temp_.mutable_gpu_diff()); 


  if(cliplimit_>0){
    int maxidx;
    caffe_gpu_absmax<Dtype>(temp_.count(),temp_.gpu_diff(),&maxidx);
    maxidx-=1;  //because the returned value is in fortran-style indexing

    BernoulliClipGrad<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->count(), D, loss_wt, maxidx, cliplimit_, temp_.gpu_diff(), breakdown, bottom[0]->mutable_gpu_diff());  //dy

    CUDA_POST_KERNEL_CHECK;
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(BernoulliLossLayer);


}  // namespace caffe
