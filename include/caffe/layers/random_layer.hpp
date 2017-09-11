#ifndef CAFFE_RANDOM_LAYER_HPP_
#define CAFFE_RANDOM_LAYER_HPP_

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
 * @brief Just generate a blob of the specified size and fill it with random 
 *        numbers, uniformly distributed between range_min_ and range_max_.
 */
template <typename Dtype>
class RandomLayer : public Layer<Dtype> {
 public:
  explicit RandomLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Random"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  vector<int> blob_shape_;
  Dtype range_min_,range_max_;

};


}  // namespace caffe

#endif  // CAFFE_RANDOM_LAYER_HPP_
                                                           
