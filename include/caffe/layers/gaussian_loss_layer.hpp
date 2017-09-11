#ifndef CAFFE_GAUSSIAN_LOSS_LAYER_HPP_
#define CAFFE_GAUSSIAN_LOSS_LAYER_HPP_

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
 * @brief Given two bottom blobs representing D-dimensional vectors mu and 
 *        sigma, and a third bottom blob representing a D-dimensional vector 
 *        x, computes a loss, log p(x|mu,sigma), where p(x|mu,sigma) is a 
 *        D-dimensional gaussian function centered at mu with width sigma.  
 *        The gaussian function here includes no correlation terms among the 
 *        axes.
 * 
 * Bottom blobs:
 *  - mu (arbitrary shape)
 *  - sigma, same shape as mu.  The absolute value is taken, and a small constant 
 *    epsilon_ is added before use in the Gaussian function to prevent 
 *    division by zero.
 *  - x, same shape as sigma and mu.  Optional.  If not provided, the loss 
 *    function is the Kullback-Liebler divergence between the Gaussian defined
 *    by mu/sigma and the standard normal distribution in D dimensions.
 * 
 * Top blobs:
 *  - loss, a scalar
 *  - the per-pattern breakdown of the loss, shape (N,1)
*/

template <typename Dtype>
class GaussianLossLayer : public Layer<Dtype> {
 public:
  explicit GaussianLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GaussianLoss"; }
  virtual inline int MinNumBottomBlobs() const { return 2; }
  virtual inline int MaxNumBottomBlobs() const { return 3; }
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

#endif  // CAFFE_GAUSSIAN_LOSS_LAYER_HPP_
                                                           
