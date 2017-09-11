#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/increment_layer.hpp"

namespace caffe {

template <typename Dtype>
void IncrementLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[0]->count(),1) << "counter blob must be a scalar";

}

template <typename Dtype>
void IncrementLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  top[0]->ReshapeLike(*(bottom[0]));

}

template <typename Dtype>
void IncrementLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* in_data=bottom[0]->cpu_data();
  Dtype* out_data=top[0]->mutable_cpu_data();
  out_data[0]=in_data[0]+1;

}

template <typename Dtype>
void IncrementLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}


#ifdef CPU_ONLY
STUB_GPU(IncrementLayer);
#endif

INSTANTIATE_CLASS(IncrementLayer);
REGISTER_LAYER_CLASS(Increment);

}  // namespace caffe
