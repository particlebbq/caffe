#include <vector>

#include "caffe/layers/gaussian_sample_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void GaussianSampleLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {


  if(bottom.size()>1){
    CHECK_EQ(bottom[0]->shape(0),bottom[1]->shape(0)) << "mu and sigma must have the same number of entries along axis 0";
    CHECK_EQ(bottom[0]->count(),bottom[1]->count()) << "mu and sigma must have the same count";
  }

  top[0]->ReshapeLike(*(bottom[0]));
  temp_.ReshapeLike(*(bottom[0]));

  if(top.size()>=2) top[1]->ReshapeLike(*(bottom[0]));

  sigma_=fmaxf(0.01,this->layer_param().gaussian_mc_param().sigma());
  cliplimit_=this->layer_param_.clip_param().cliplimit();

}


template <typename Dtype>
void GaussianSampleLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  top[0]->ReshapeLike(*(bottom[0]));
  temp_.ReshapeLike(*(bottom[0]));

  if(top.size()>=2) top[1]->ReshapeLike(*(bottom[0]));

}

template <typename Dtype>
void GaussianSampleLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {


  const Dtype* mu=bottom[0]->cpu_data();
  const Dtype* sigma=NULL;
  if(bottom.size()>1) sigma=bottom[1]->cpu_data();

  int N=bottom[0]->shape(0);
  int D=bottom[0]->count(1);

  Dtype* sample=top[1]->mutable_cpu_data();
  for(int item=0;item<N;item++){
    for(int idim=0;idim<D;idim++){
      Dtype sig=sigma_;
      if(sigma) sig=fmaxf(0.01,fabs(sigma[item*D+idim]));
      caffe_rng_gaussian<Dtype>(1,mu[item*D+idim],sig,&(sample[item*D+idim]));
    }
  }

  if(top.size()>=2){
    if(bottom.size()>1){
      caffe_copy(bottom[1]->count(),
                 bottom[1]->cpu_data(),
                 top[1]->mutable_cpu_data());
    } else {
      caffe_set(top[1]->count(),
                sigma_,
                top[1]->mutable_cpu_data());
    }
  }

}

template <typename Dtype>
void GaussianSampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){


  const Dtype* mu=bottom[0]->cpu_data();
  const Dtype* sigma=NULL;
  if(bottom.size()>1) sigma=bottom[1]->cpu_data();
  const Dtype* sample=top[0]->cpu_data();
  const Dtype* samp_diff=top[0]->cpu_diff();

  Dtype* dmu=bottom[0]->mutable_cpu_diff();
  Dtype* dsigma=NULL;
  if(bottom.size()>1) dsigma=bottom[1]->mutable_cpu_diff();

  int N=bottom[0]->shape(0);
  int D=bottom[0]->shape(1);

  Dtype maxgrad=0;
  for(int item=0;item<N;item++){
    for(int idim=0;idim<D;idim++){
      dmu[item*D+idim]=samp_diff[item*D+idim]; 
      if(sigma) {
        Dtype sig=fmaxf(0.01,fabs(sigma[item*D+idim]));
        dsigma[item*D+idim]=samp_diff[item*D+idim]*(sample[item*D+idim]-mu[item*D+idim])/sig;
        if(fabs(dsigma[item*D+idim])>maxgrad || fabs(dmu[item*D+idim])>maxgrad) maxgrad=fmaxf(fabs(dsigma[item*D+idim]),fabs(dmu[item*D+idim]));
      }
    }
  }

  if(cliplimit_>0 && maxgrad>cliplimit_ && sigma!=NULL){
    for(int i=0;i<bottom[0]->count();i++){
      dmu[i]*=cliplimit_/maxgrad;
      dsigma[i]*=cliplimit_/maxgrad;
    }
  }

};

#ifdef CPU_ONLY
STUB_GPU(GaussianSampleLayer);
#endif

INSTANTIATE_CLASS(GaussianSampleLayer);
REGISTER_LAYER_CLASS(GaussianSample);

}  // namespace caffe
