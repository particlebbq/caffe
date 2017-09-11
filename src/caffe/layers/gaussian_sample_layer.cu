#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/gaussian_sample_layer.hpp"

namespace caffe {


template <typename Dtype>
__global__ void GaussianSampleBackward(const int nthreads, 
    const Dtype* mu, const Dtype* sigma, const Dtype* samp_diff, 
    const Dtype* sample, const int D, const int N, const bool have_dsigma, Dtype* dmu, Dtype* dsigma) {
  CUDA_KERNEL_LOOP(index, nthreads) {

    if(index>=nthreads) return;

    int item = index/D;
    int idim   = index%D;

    dmu[item*D+idim]=samp_diff[item*D+idim];
    if(have_dsigma){
      Dtype sig=fmaxf(0.01,fabs(sigma[item*D+idim]));
      dsigma[item*D+idim]=samp_diff[item*D+idim]
                         *(sample[item*D+idim]-mu[item*D+idim])/sig;

    }

  }

}




template <typename Dtype>
__global__ void GaussClipGrad(const int nthreads, const int D,
    const int mu_idx, const int sigma_idx, const Dtype maxval, 
    const Dtype* in_mu, const Dtype* in_sigma, Dtype* out_mu, 
    Dtype* out_sigma) {
  CUDA_KERNEL_LOOP(index, nthreads) {


    if(index>=nthreads) return;

    Dtype gradmax=fmaxf(fabs(in_mu[mu_idx]),fabs(in_sigma[sigma_idx]));
    Dtype scalefac=1.;
    if(gradmax>maxval) scalefac=maxval/gradmax;

    out_mu[index]=scalefac*in_mu[index];
    out_sigma[index]=scalefac*in_sigma[index];

  }
}




template <typename Dtype>
void GaussianSampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


  const Dtype* mu=bottom[0]->gpu_data();
  const Dtype* sigma=NULL;
  if(bottom.size()>1) sigma=bottom[1]->gpu_data();
  Dtype* sample=top[0]->mutable_gpu_data();

  if(sigma){
    caffe_gpu_rng_gaussian<Dtype>(bottom[0]->count(),Dtype(0.),Dtype(1.),sample);
    caffe_gpu_mul<Dtype>(bottom[0]->count(),sample,sigma,sample);
  } else {
    caffe_gpu_rng_gaussian<Dtype>(bottom[0]->count(),Dtype(0.),Dtype(sigma_),sample);
  }
  caffe_gpu_add<Dtype>(bottom[0]->count(),sample,mu,sample);

  CUDA_POST_KERNEL_CHECK;


  if(top.size()>=2){
    if(bottom.size()>1){
      caffe_copy(bottom[1]->count(),
                 bottom[1]->gpu_data(),
                 top[1]->mutable_gpu_data());
    } else {
      caffe_set(top[1]->count(),
                sigma_,
                top[1]->mutable_cpu_data());
    }
  }


}


template <typename Dtype>
void GaussianSampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* mu=bottom[0]->gpu_data();
  const Dtype* sigma=NULL;
  if(bottom.size()>1) sigma=bottom[1]->gpu_data();
  const Dtype* sample=top[0]->gpu_data();
  const Dtype* samp_diff=top[0]->gpu_diff();
  Dtype* dmu=bottom[0]->mutable_gpu_diff();
  Dtype* dsigma=NULL;
  if(bottom.size()>1) dsigma=bottom[1]->mutable_gpu_diff();

  int N=bottom[0]->shape(0);
  int D=bottom[0]->shape(1);

  GaussianSampleBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
      bottom[0]->count(), mu, sigma, samp_diff, sample, D, N, dsigma!=NULL,
      temp_.mutable_gpu_data(), temp_.mutable_gpu_diff());  //last args are dmu, dsigma

  CUDA_POST_KERNEL_CHECK;

  if(sigma==NULL) return;

  if(cliplimit_>0){
    int maxgrad_mu_idx,maxgrad_sigma_idx;
    caffe_gpu_absmax<Dtype>(temp_.count(),temp_.gpu_data(),&maxgrad_mu_idx);
    caffe_gpu_absmax<Dtype>(temp_.count(),temp_.gpu_diff(),&maxgrad_sigma_idx);
    maxgrad_mu_idx-=1;    //correct for fortran-style indexing 
    maxgrad_sigma_idx-=1; 
  
    GaussClipGrad<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->count(), D, maxgrad_mu_idx, maxgrad_sigma_idx, cliplimit_,
        temp_.gpu_data(), temp_.gpu_diff(), dmu, dsigma); 

    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GaussianSampleLayer);


}  // namespace caffe
