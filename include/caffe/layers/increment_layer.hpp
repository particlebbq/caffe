#ifndef CAFFE_INCREMENT_LAYER_HPP_
#define CAFFE_INCREMENT_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
/**
 * @brief Given a scalar blob (count==1), add one to its contents.  Backward
 *        is a no-op.
 *
 * Bottom blobs:
 *   - counter
 * Top blobs:
 *   - counter incremented by 1
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class IncrementLayer : public Layer<Dtype> {
 public:
  explicit IncrementLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Increment"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; } 
  virtual inline int ExactNumTopBlobs() const { return 1; }   

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe

#endif  // CAFFE_INCREMENT_LAYER_HPP_

