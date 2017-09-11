#ifndef CAFFE_BERNOULLI_LOSS_LAYER_HPP_
#define CAFFE_BERNOULLI_LOSS_LAYER_HPP_

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
 *        probabilities y, and a second bottom blob representing a sample x, 
 *        computes a loss, log p(x|y) = sum_{D}[ x*log(y)+(1-x)*log(1-y) ]
 * 
 *        Note that the y values are not required to sum to 1; each entry in y 
 *        is treated as its own independent probability.
 * 
 * Bottom blobs:
 *  - y, probability vector (arbitrary shape)
 *  - x, same shape as y.
 * 
 * Top blobs:
 *  - loss, a scalar
 *  - optional: the per-item breakdown of the loss, shape (N,1)
*/

template <typename Dtype>
class BernoulliLossLayer : public Layer<Dtype> {
 public:
  explicit BernoulliLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BernoulliLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int MinNumTopBlobs() const { return 1; }
  virtual inline int MaxNumTopBlobs() const { return 2; }

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
  Dtype cliplimit_;
};


}  // namespace caffe

#endif  // CAFFE_BERNOULLI_LOSS_LAYER_HPP_
                                                           
