#ifndef CAFFE_BERNOULLI_SAMPLE_LAYER_HPP_
#define CAFFE_BERNOULLI_SAMPLE_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


/**
 * @brief Given a bottom blob representing a D-dimensional vector of 
 *        probabilities p_{i}, randomly draw a 0 or 1 for each entry,
 *        choosing 0 with probability p_{i} and 1 with probability 1-p_{i}.
 * 
 * Bottom blobs:
 *  - p, probability vector (arbitrary shape)
 * 
 * Top blobs:
 *  - x, a random sample from p
*/

template <typename Dtype>
class BernoulliSampleLayer : public Layer<Dtype> {
 public:
  explicit BernoulliSampleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BernoulliSample"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> temp_;
};


}  // namespace caffe

#endif  // CAFFE_BERNOULLI_SAMPLE_LAYER_HPP_
                                                           
