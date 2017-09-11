#ifndef CAFFE_GLIMPSE_CROPPING_LAYER_HPP_
#define CAFFE_GLIMPSE_CROPPING_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
/**
 * @brief crops an image to a small region around a given location, for use as 
 *        input to a glimpse network
 *
 * Bottom blobs:
 *  - the full image
 *  - a location tuple describing where to crop from
 *  - (optional) if the image is a variable-sized image embedded in a larger
 *    fixed size blob, the third bottom blob provides the boundaries of the 
 *    embedded image
 * 
 * Top blobs:
 *  - the cropped image
 *
 */
template <typename Dtype>
class GlimpseCroppingLayer : public Layer<Dtype> {
 public:
  explicit GlimpseCroppingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GlimpseCropping"; }
  virtual inline int MinNumBottomBlobs() const { return 2; }  
  virtual inline int MaxNumBottomBlobs() const { return 3; }  
  virtual inline int ExactNumTopBlobs() const { return 1; }  

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  //the cropped region is a square with crop_size_ pixels on a side
  int crop_size_;  
  int channels_;
  int height_, width_;
  bool do_downsamp_;
};

}  // namespace caffe

#endif  // CAFFE_GLIMPSE_CROPPING_LAYER_HPP_

