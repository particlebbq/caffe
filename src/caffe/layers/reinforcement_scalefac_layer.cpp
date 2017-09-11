#include <vector>

#include "caffe/layers/reinforcement_scalefac_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void ReinforcementScalefacLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  lambda_=this->layer_param_.reinforcement_param().lambda();

}

template <typename Dtype>
void ReinforcementScalefacLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[0]->shape(0),bottom[1]->shape(0))
      << "Bottom blobs must have the same number of entries along axis 0";
  CHECK_EQ(bottom[1]->shape(0),bottom[1]->count())
      << "baseline blob must have shape (N,)";

  top[0]->ReshapeLike(*(bottom[0]));
  top[1]->ReshapeLike(*(bottom[1])); 
  if(top.size()>=3) top[2]->ReshapeLike(*(bottom[1]));
}

template <typename Dtype>
void ReinforcementScalefacLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  caffe_copy(bottom[0]->count(),
             bottom[0]->cpu_data(),
             top[0]->mutable_cpu_data());

}

template <typename Dtype>
void ReinforcementScalefacLayer<Dtype>::Backward_cpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, 
      const vector<Blob<Dtype>*>& bottom){


  Dtype* out_diff=bottom[0]->mutable_cpu_diff();  
  const Dtype* in_diff=top[0]->cpu_diff();
  const Dtype* b=bottom[1]->cpu_data();
  const Dtype* R=top[0]->cpu_diff();  
  const Dtype* curr=NULL;
  if(top.size()>2) curr=top[2]->cpu_diff();
  int C=bottom[0]->count(1);

  for(int item=0;item<bottom[0]->shape(0);item++){
    Dtype curr_weight=1.;
    if(curr) curr_weight=curr[item];
    for(int ic=0;ic<C;ic++){
      out_diff[item*C+ic]=lambda_*curr_weight*(R[item]-b[item])*in_diff[item*C+ic]; 
    }
  }

}

#ifdef CPU_ONLY
STUB_GPU(ReinforcementScalefacLayer);
#endif

INSTANTIATE_CLASS(ReinforcementScalefacLayer);
REGISTER_LAYER_CLASS(ReinforcementScalefac);

}  // namespace caffe
