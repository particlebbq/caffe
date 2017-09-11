#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/gaussian_loss_layer.hpp"

namespace caffe {



template <typename Dtype>
__global__ void GaussianLossForward(const int nthreads,
    const Dtype* x, const Dtype* mu, const Dtype* sigma, const int D, 
    const int N, const int bottom_size, Dtype* loss_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {

    if(index>=nthreads) return;

    int item = index/D;
    int idim   = index%D;

    Dtype logprob=0;
    Dtype sig=fmaxf(0.01,fabs(sigma[item*D+idim]));
    if(bottom_size>2){
      logprob-=-log(sqrt(2*3.14159265*sig*sig))
              -(pow(x[item*D+idim]-mu[item*D+idim],2)/(2*pow(sig,2)) );
    } else {
      logprob-=0.5*(1+log(sig*sig)
                        -(mu[item*D+idim]*mu[item*D+idim])
                        -(sig*sig));
    }
    loss_data[index]=logprob/N;
  }

}



template <typename Dtype>
__global__ void GaussianLossBackward(const int nthreads, const Dtype* x, 
    const Dtype* mu, const Dtype* sigma, const int D, const int N, 
    const int bottom_size, Dtype* dmu, Dtype* dsigma) {
  CUDA_KERNEL_LOOP(index, nthreads) {

    if(index>=nthreads) return;

    int item = index/D;
    int idim   = index%D;

    Dtype sig=fmaxf(0.01,fabs(sigma[item*D+idim]));
    if(bottom_size>2){
      dmu[item*D+idim]=(-1*(x[item*D+idim]-mu[item*D+idim])/(sig*sig))/N;
      dsigma[item*D+idim]=
         (1/sig - pow(x[item*D+idim]-mu[item*D+idim],2)/pow(sig,3))/N;

    } else {

      dmu[item*D+idim]=mu[item*D+idim]/N;
      dsigma[item*D+idim]=(-1*(1/sig - sig))/N;

    }

  }

}




template <typename Dtype>
__global__ void GaussClipGrad(const int nthreads, const int D,
    const Dtype* loss_wt, const int mu_idx, const int sigma_idx, 
    const Dtype maxval, const Dtype* in_mu, const Dtype* in_sigma, 
    const Dtype* breakdown, Dtype* out_mu, Dtype* out_sigma) {
  CUDA_KERNEL_LOOP(index, nthreads) {


    if(index>=nthreads) return;

    int item=index/D;

    Dtype gradmax=fmaxf(fabs(in_mu[mu_idx]),fabs(in_sigma[sigma_idx]));
    Dtype scalefac=1.;
    if(maxval>0 && gradmax>maxval) scalefac=maxval/gradmax;
    if(breakdown){
      scalefac*=breakdown[item];
    } else {
      scalefac*=loss_wt[0];
    }

    out_mu[index]=scalefac*in_mu[index];
    out_sigma[index]=scalefac*in_sigma[index];

  }
}




template <typename Dtype>
void GaussianLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


  const Dtype* mu=bottom[0]->gpu_data();
  const Dtype* sigma=bottom[1]->gpu_data();
  const Dtype* x=NULL;
  if(bottom.size()>2){
    x=bottom[2]->gpu_data();
  }
  Dtype* loss=top[0]->mutable_cpu_data();
  Dtype* breakdown=NULL;
  if(top.size()>=2){
    breakdown=top[1]->mutable_cpu_data();
  }


  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();

  int N=bottom[0]->shape(0);
  int D=bottom[0]->count(1);

  loss[0]=0;

    GaussianLossForward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->count(), x, mu, sigma, D, N, bottom.size(), loss_data);

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
void GaussianLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


  const Dtype* mu=bottom[0]->gpu_data();
  const Dtype* sigma=bottom[1]->gpu_data();
  const Dtype* x=NULL;
  if(bottom.size()>2){
    x=bottom[2]->gpu_data();
  }
  const Dtype* loss_wt=top[0]->gpu_diff();
  const Dtype* breakdown=NULL;
  if(top.size()>=2) {
    breakdown=top[1]->gpu_diff();
  }

  Dtype* dmu=bottom[0]->mutable_gpu_diff();
  Dtype* dsigma=bottom[1]->mutable_gpu_diff();

  int N=bottom[0]->shape(0);
  int D=bottom[0]->shape(1);


  if(breakdown){  //we will apply loss weights from breakdown, not global loss, during clip grad later
    GaussianLossBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->count(), x, mu, sigma, D, N, bottom.size(), 
        temp_.mutable_gpu_data(), temp_.mutable_gpu_diff());  //last args are dmu, dsigma
  } else {
    GaussianLossBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->count(), x, mu, sigma, D, N, bottom.size(),
        temp_.mutable_gpu_data(), temp_.mutable_gpu_diff());  //last args are dmu, dsigma
  }

  int maxgrad_mu_idx,maxgrad_sigma_idx;
  caffe_gpu_absmax<Dtype>(temp_.count(),temp_.gpu_data(),&maxgrad_mu_idx);
  caffe_gpu_absmax<Dtype>(temp_.count(),temp_.gpu_diff(),&maxgrad_sigma_idx);
  maxgrad_mu_idx-=1;    //correct for fortran-style indexing 
  maxgrad_sigma_idx-=1; 

  GaussClipGrad<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
      bottom[0]->count(), D, loss_wt, maxgrad_mu_idx, maxgrad_sigma_idx, 
      cliplimit_,temp_.gpu_data(), temp_.gpu_diff(), breakdown,
      bottom[0]->mutable_gpu_diff(), bottom[1]->mutable_gpu_diff());  //dmu, dsigma


  CUDA_POST_KERNEL_CHECK;

}

INSTANTIATE_LAYER_GPU_FUNCS(GaussianLossLayer);


}  // namespace caffe
