#ifndef CAFFE_REINFORCEMENT_SCALEFAC_LAYER_HPP_
#define CAFFE_REINFORCEMENT_SCALEFAC_LAYER_HPP_

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
 * @brief In some reinforcement learning settings, it is necessary to scale
 *        a gradient by a factor lambda*(R-b), where lambda is a 
 *        hyperparameter, R is a reward, and b is a "baseline" estimate of that
 *        reward.  This layer computes that scale factor and applies it to 
 *        a gradient flowing along some variable X during the backward pass.
 * 
 * Bottom blobs:
 *   - X:  the quantity whose gradient needs to be scaled by lambda*(R-b)
 *   - b:  the baseline estimate.  Shape (N,), where N is the same as the size
 *     of axis 0 in the X blob.
 * Top blobs:
 *   - X:  same as bottom blob; forward pass just copies bottom[0] over to this
 *   - reward:  since the reward generally is calculated at the end of the 
 *     forward pass, it needs to be communicated back to this layer via the 
 *     diff block of a blob during the backward pass.  No values are placed in 
 *     the data block of this blob in the forward pass.
 *   - curriculum cutoff flag (optional). Same shape as b.  if present, this 
 *     will be multiplied into the scale factor.  The intention (which is not 
 *     enforced) is that this will be a list of 0/1 values so that, for 
 *     example in a recurrent model, if the network has failed to make a 
 *     correct prediction in the previous timestep, the gradients from this
 *     timestep will not compound the error.  As with the reward, this cannot
 *     always be calculated during the forward pass, so it must be passed to 
 *     this layer as the diff block of one of the top blobs.
 */
template <typename Dtype>
class ReinforcementScalefacLayer : public Layer<Dtype> {
 public:
  explicit ReinforcementScalefacLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ReinforcementScalefac"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int MinNumTopBlobs() const { return 2; }  
  virtual inline int MaxNumTopBlobs() const { return 3; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype lambda_;
};


}  // namespace caffe

#endif  // CAFFE_REINFORCEMENT_SCALEFAC_LAYER_HPP_
                                                           
