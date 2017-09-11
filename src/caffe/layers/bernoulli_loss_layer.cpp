#include <vector>

#include "caffe/layers/bernoulli_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void BernoulliLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  cliplimit_=this->layer_param_.clip_param().cliplimit();

  CHECK_EQ(bottom[0]->shape(0),bottom[1]->shape(0)) 
      << "x and y must have the same number of entries along axis 0";
  CHECK_EQ(bottom[0]->count(),bottom[1]->count()) 
      << "x and y must have the same count";

  temp_.ReshapeLike(*(bottom[0]));

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
void BernoulliLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  temp_.ReshapeLike(*(bottom[0]));

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
void BernoulliLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {


  const Dtype* y=bottom[0]->cpu_data();
  const Dtype* x=bottom[1]->cpu_data();
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
      Dtype p=x[item*D+idim]*log(fmaxf(0.01,y[item*D+idim]))
             +(1-x[item*D+idim])*log(fmaxf(0.01,1-y[item*D+idim]));
      logprob+=p;
      breakdown[item]-=p/N;
    }
    loss[0]-=logprob/N;
  }

}

template <typename Dtype>
void BernoulliLossLayer<Dtype>::Backward_cpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, 
      const vector<Blob<Dtype>*>& bottom){


  const Dtype* y=bottom[0]->cpu_data();
  const Dtype* x=bottom[1]->cpu_data();
  const Dtype* loss_wt=top[0]->cpu_diff();
  const Dtype* breakdown=NULL;
  if(top.size()>=2){
    breakdown=top[1]->mutable_cpu_diff();
  }

  Dtype* dy=bottom[0]->mutable_cpu_diff();

  int N=bottom[0]->shape(0);
  int D=bottom[0]->shape(1);

  Dtype maxgrad=0;
  for(int item=0;item<N;item++){
    Dtype wgt=loss_wt[0];
    if(breakdown){
      wgt=breakdown[item];
    }
    for(int idim=0;idim<D;idim++){
      dy[item*D+idim]=-wgt*(x[item*D+idim]/fmaxf(0.01,y[item*D+idim])
                      -(1-x[item*D+idim])/(fmaxf(0.01,1-y[item*D+idim])))/N; 
      if(fabs(dy[item*D+idim])>maxgrad) maxgrad=fabs(dy[item*D+idim]);

    }
  }

  if(cliplimit_>0 && maxgrad>cliplimit_){
    for(int i=0;i<bottom[0]->count();i++){
      dy[i]*=cliplimit_/maxgrad;
    }
  }

};

INSTANTIATE_CLASS(BernoulliLossLayer);
REGISTER_LAYER_CLASS(BernoulliLoss);

}  // namespace caffe
