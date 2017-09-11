#include <vector>

#include "caffe/layers/input_layer.hpp"

namespace caffe {

template <typename Dtype>
void InputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_top = top.size();
  const InputParameter& param = this->layer_param_.input_param();
  const int num_shape = param.shape_size();
  CHECK(num_shape == 0 || num_shape == 1 || num_shape == num_top)
      << "Must specify 'shape' once, once per top blob, or not at all: "
      << num_top << " tops vs. " << num_shape << " shapes.";
  if (num_shape > 0) {
    for (int i = 0; i < num_top; ++i) {
      const int shape_index = (param.shape_size() == 1) ? 0 : i;
      top[i]->Reshape(param.shape(shape_index));
    }
  }

  this->blobs_.resize(1);
  this->blobs_[0].reset(new Blob<Dtype>());  //dummy so we will get need_backward
  vector<int> dummy_shape;
  dummy_shape.push_back(1);
  this->blobs_[0]->Reshape(dummy_shape);
  this->blobs_[0]->mutable_cpu_data();
  this->param_propagate_down_.resize(this->blobs_.size(), true);


}

INSTANTIATE_CLASS(InputLayer);
REGISTER_LAYER_CLASS(Input);

}  // namespace caffe
