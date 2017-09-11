#include <vector>

#include "caffe/layers/zero_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ZeroLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const ZeroParameter& zero_param = this->layer_param_.zero_param();
  blob_shape_.clear();
  CHECK_GT(zero_param.shape_size(),0) << "ZeroLayer needs to know what shape blob to make";
  for(int i=0;i<zero_param.shape_size();i++) blob_shape_.push_back(zero_param.shape(i));

  top[0]->Reshape(blob_shape_);
  Dtype* top_data=top[0]->mutable_cpu_data();
  for(int i=0;i<top[0]->count();i++) top_data[i]=0;

}

template <typename Dtype>
void ZeroLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 
}

template <typename Dtype>
void ZeroLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
}

#ifdef CPU_ONLY
STUB_GPU(ZeroLayer);
#endif

INSTANTIATE_CLASS(ZeroLayer);
REGISTER_LAYER_CLASS(Zero);

}  // namespace caffe
