#include <vector>

#include "caffe/layers/gaussian_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void GaussianLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

/*
 * Bottom blobs:
 *  - mu (arbitrary shape)
 *  - sigma, same shape as mu.  The absolute value is taken, and a small constant 
 *    epsilon_ is added before use in the Gaussian function to prevent 
 *    division by zero.
 *  - x, same shape as sigma and mu.  Optional.
 * 
 * Top blobs:
 *  - loss, a scalar
*/

  cliplimit_=this->layer_param_.clip_param().cliplimit();

  CHECK_EQ(bottom[0]->shape(0),bottom[1]->shape(0)) << "mu and sigma must have the same number of entries along axis 0";
  if(bottom.size()>2){
    CHECK_EQ(bottom[0]->shape(0),bottom[2]->shape(0)) << "mu and x must have the same number of entries along axis 0";
  }

  CHECK_EQ(bottom[0]->count(),bottom[1]->count()) << "mu and sigma must have the same count";
  if(bottom.size()>2){
    CHECK_EQ(bottom[0]->count(),bottom[2]->count()) << "mu and x must have the same count";
  }

  vector<int> loss_shape;
  loss_shape.push_back(1);
  top[0]->Reshape(loss_shape);
  temp_.ReshapeLike(*(bottom[0]));

  if(top.size()>=2){
    vector<int> breakdown_shape;
    breakdown_shape.push_back(bottom[0]->shape(0));
    breakdown_shape.push_back(1);
    top[1]->Reshape(breakdown_shape);
  }

  this->SetLossWeights(top);
}


template <typename Dtype>
void GaussianLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  vector<int> loss_shape;
  loss_shape.push_back(1);
  top[0]->Reshape(loss_shape);

  if(top.size()>=2){
    vector<int> breakdown_shape;
    breakdown_shape.push_back(bottom[0]->shape(0));
    breakdown_shape.push_back(1);
    top[1]->Reshape(breakdown_shape);
  }

}

template <typename Dtype>
void GaussianLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {


  const Dtype* mu=bottom[0]->cpu_data();
  const Dtype* sigma=bottom[1]->cpu_data();
  const Dtype* x=NULL;
  if(bottom.size()>2){
    x=bottom[2]->cpu_data();
  }
  Dtype* loss=top[0]->mutable_cpu_data();
  Dtype* breakdown=NULL;
  if(top.size()>=2){
    breakdown=top[1]->mutable_cpu_data();
  }

  int N=bottom[0]->shape(0);
  int D=bottom[0]->count(1);

  loss[0]=0;
  for(int item=0;item<N;item++){
    Dtype logprob=0;
    if(breakdown) breakdown[item]=0.;
    for(int idim=0;idim<D;idim++){
      Dtype sig=fmaxf(0.01,fabs(sigma[item*D+idim]));
      if(bottom.size()>2){
        Dtype p=-log(sqrt(2*M_PI*sig*sig))
                -(pow(x[item*D+idim]-mu[item*D+idim],2)/(2*pow(sig,2)) );
        logprob-=p; 
        breakdown[item]-=p/N;
      } else {
        
        Dtype p=0.5*(1+log(sig*sig)
                          -(mu[item*D+idim]*mu[item*D+idim])
                          -(sig*sig));
        logprob-=p;
        breakdown[item]-=p/N;
      }
    }
    loss[0]+=logprob/N;
  }

}

template <typename Dtype>
void GaussianLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){


  const Dtype* mu=bottom[0]->cpu_data();
  const Dtype* sigma=bottom[1]->cpu_data();
  const Dtype* x=NULL;
  if(bottom.size()>2){
    x=bottom[2]->cpu_data();
  }
  const Dtype* loss_wt=top[0]->cpu_diff();
  const Dtype* breakdown=NULL;
  if(top.size()>=2){
    breakdown=top[1]->mutable_cpu_diff();
  }


  Dtype* dmu=bottom[0]->mutable_cpu_diff();
  Dtype* dsigma=bottom[1]->mutable_cpu_diff();

  int N=bottom[0]->shape(0);
  int D=bottom[0]->shape(1);

  Dtype maxgrad=0;
  for(int item=0;item<N;item++){
    Dtype wgt=loss_wt[0];
    if(breakdown){
      wgt=breakdown[item];
    }
    for(int idim=0;idim<D;idim++){
      Dtype sig=fmaxf(0.01,fabs(sigma[item*D+idim]));

      if(bottom.size()>2){
        //logprob+= (log(sqrt(2*pi*sig^2))+(x-mu)^2/(2*sig^2))
        //
        //so:
        //  * d(logprob)/dmu=-(x-mu)/sigma^2
        //  * d(logprob)/dsigma=1/sigma - (x-mu)^2/sigma^3
        //...for each item & idim, scaled by loss weight
        //xms_avg+=((x[item*D+idim]-mu[item*D+idim])/sig)/N*D;

        dmu[item*D+idim]=(-wgt*(x[item*D+idim]-mu[item*D+idim])/(sig*sig))/N;
        dsigma[item*D+idim]=wgt
                 *(1/sig - pow(x[item*D+idim]-mu[item*D+idim],2)/pow(sig,3))/N;
        if(fabs(dsigma[item*D+idim])>maxgrad || fabs(dmu[item*D+idim])>maxgrad) 
            maxgrad=fmaxf(fabs(dsigma[item*D+idim]),fabs(dmu[item*D+idim]));

      } else {
        //logprob+= -1*(0.5*(1+log(sigma^2)-mu^2-sigma^2))
        //
        //so from this term:
        //  * d(logprob)/dmu=mu
        //  * d(logprob)/dsigma=-1*(1/sigma - sigma)
        //
        dmu[item*D+idim]=(1*wgt*mu[item*D+idim])/N; 
        dsigma[item*D+idim]=(-1*wgt*(1/sig - sig))/N;

        if(fabs(dsigma[item*D+idim])>maxgrad || fabs(dmu[item*D+idim])>maxgrad) 
          maxgrad=fmaxf(fabs(dsigma[item*D+idim]),fabs(dmu[item*D+idim]));

      }
    }
  }

  if(cliplimit_>0 && maxgrad>cliplimit_){
    for(int i=0;i<bottom[0]->count();i++){
      dmu[i]*=cliplimit_/maxgrad;
      dsigma[i]*=cliplimit_/maxgrad;
    }
  }

};

#ifdef CPU_ONLY
STUB_GPU(GaussianLossLayer);
#endif

INSTANTIATE_CLASS(GaussianLossLayer);
REGISTER_LAYER_CLASS(GaussianLoss);

}  // namespace caffe
