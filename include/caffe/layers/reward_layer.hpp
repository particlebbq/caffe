#ifndef CAFFE_REWARD_LAYER_HPP_
#define CAFFE_REWARD_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief For use in reinforcement learning.  Given some set of predictions and 
 *        labels, compute a reward function (1 if the highest-probability 
 *        predicted label is correct, 0 otherwise) for each predicted label.  
 *        Fill the diff of the prediction blob with the results, and the top 
 *        blob with the sum.
 * 
 * Bottom blobs:
 *   - predictions:  shape (N*M,C,T) where C is the number of channels and T is 
 *     the number of timesteps, and M is an integer (like the number of toyMC 
 *     tosses for each input)
 *   - labels:  shape (N,t), where the values are integers ranging from 0 to 
 *     C-1 (inclusive) and T is evenly divisble by the number of targets t
 *   - rewards:  shape (N*M,T).  This input is ignored during the forward pass, 
 *     but during the backward pass the calculated reward gets put into the 
 *     diff part of this blob
 *   - curriculum cutoff (optional):  shape(N*M,T).  Ignored during forward; 
 *     curriculum flags placed in diff during backward.  A value of 1 means 
 *     that all classifications before this point are correct (i.e. you got 
 *     the reward); otherwise, 0.
 */
template <typename Dtype>
class RewardLayer : public Layer<Dtype> {
 public:
  explicit RewardLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {max_correct_=-1e30; min_pred_=1e30;}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Reward"; }

  //the prediction, label, and reward blobs
  virtual inline int MinNumBottomBlobs() const { return 3; }

  //optional: curriculum cutoff
  virtual inline int MaxNumBottomBlobs() const { return 4; } 

  //the sum of the rewards obtained in this batch
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool per_target_rewards_;
  Blob<Dtype> diff_;
  int num_targets_;
  Dtype max_correct_, min_pred_;
};

}  // namespace caffe

#endif  // CAFFE_REWARD_LAYER_HPP_
