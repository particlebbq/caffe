#include <vector>

#include "caffe/layers/random_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RandomLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const RandomParameter& rand_param = this->layer_param_.rand_param();
  blob_shape_.clear();
  CHECK_GT(rand_param.shape_size(),0) 
    << "RandomLayer needs to know what shape blob to make";
  for(int i=0;i<rand_param.shape_size();i++) 
    blob_shape_.push_back(rand_param.shape(i));

  range_min_=rand_param.range_min();
  range_max_=rand_param.range_max();

  top[0]->Reshape(blob_shape_);
  Dtype* top_data=top[0]->mutable_cpu_data();
  for(int i=0;i<top[0]->count();i++) top_data[i]=0;

}

template <typename Dtype>
void RandomLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  caffe_rng_uniform<Dtype>(top[0]->count(),
                           range_min_,
                           range_max_,
                           top[0]->mutable_cpu_data());

}

template <typename Dtype>
void RandomLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
}

#ifdef CPU_ONLY
STUB_GPU(RandomLayer);
#endif

INSTANTIATE_CLASS(RandomLayer);
REGISTER_LAYER_CLASS(Random);

}  // namespace caffe
