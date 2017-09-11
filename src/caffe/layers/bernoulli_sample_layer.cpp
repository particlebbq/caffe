#include <vector>

#include "caffe/layers/bernoulli_sample_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void BernoulliSampleLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  top[0]->ReshapeLike(*(bottom[0]));
  temp_.ReshapeLike(*(bottom[0]));

}


template <typename Dtype>
void BernoulliSampleLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  top[0]->ReshapeLike(*(bottom[0]));
  temp_.ReshapeLike(*(bottom[0]));

}

template <typename Dtype>
void BernoulliSampleLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {


  const Dtype* p=bottom[0]->cpu_data();

  int N=bottom[0]->shape(0);
  int D=bottom[0]->count(1);

  Dtype* sample=top[0]->mutable_cpu_data();
  for(int item=0;item<N;item++){
    for(int idim=0;idim<D;idim++){
      Dtype rand;
      caffe_rng_uniform<Dtype>(1,0.,1.,&rand);
      if(rand>p[item*D+idim]) {
        sample[item*D+idim]=0;
      } else {
        sample[item*D+idim]=1;
      }
    }
  }


}

template <typename Dtype>
void BernoulliSampleLayer<Dtype>::Backward_cpu(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, 
      const vector<Blob<Dtype>*>& bottom){


};

INSTANTIATE_CLASS(BernoulliSampleLayer);
REGISTER_LAYER_CLASS(BernoulliSample);

}  // namespace caffe
