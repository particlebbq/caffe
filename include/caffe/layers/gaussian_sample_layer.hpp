#ifndef CAFFE_GAUSSIAN_SAMPLE_LAYER_HPP_
#define CAFFE_GAUSSIAN_SAMPLE_LAYER_HPP_

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
 *        sigma, draw a random sample from a D-dimensional gaussian 
 *        parameterized by mu and sigma.  Propagates gradients back from 
 *        the sample to mu and sigma using the reparameterization trick, 
 *        i.e. regarding the sample as z=mu+x*sigma, where x is drawn from
 *        a standard normal distribution.
 * 
 * Bottom blobs:
 *  - mu (arbitrary shape)
 *  - sigma (optional), same shape as mu.  If not provided, a constant value
 *    taken from the .prototxt will be used instead. The absolute value is 
 *    taken, and a small constant epsilon is added before use in the Gaussian 
 *    function to prevent division by zero.
 * 
 * Top blobs:
 *  - x', a random sample from the D-dimensional distribution 
 *    gaus(mu,sigma)
 *  - sigma (optional).  Some downstream things (like GaussianLossLayer) 
 *    need to have a mu and a sigma as inputs.  If a sigma blob is provided
 *    in this layer's bottom vector, it is copied over to this; otherwise, 
 *    this is reshaped to match mu and filled with the constant value taken
 *    from the .prototxt.
*/

template <typename Dtype>
class GaussianSampleLayer : public Layer<Dtype> {
 public:
  explicit GaussianSampleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GaussianSample"; }
  virtual inline int MinNumBottomBlobs() const { return 1; }
  virtual inline int MaxNumBottomBlobs() const { return 2; }
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
  Dtype sigma_;
  Dtype cliplimit_;

};


}  // namespace caffe

#endif  // CAFFE_GAUSSIAN_SAMPLE_LAYER_HPP_
                                                           
