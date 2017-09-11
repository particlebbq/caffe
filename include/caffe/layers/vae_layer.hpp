#ifndef CAFFE_VAE_LAYER_HPP_
#define CAFFE_VAE_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"

namespace caffe {

/**
 * @brief Implements a simple variational auto-encoder, a la arXiv:1312.6114.
 * 
 * Bottom blobs: 
 *   - a data point x
 * Top blobs:
 *   - the loss function for this batch, sum(log p(x,z) - log q(z|x))
 *   - a sample z from the latent variables, drawn from q(z|x)
 *   - optional: a sample x' conditioned on the latent variable sample, i.e. 
 *     drawn from p(x',z)
 * 
 * Note that the p(x,z) in the loss function takes the x in the bottom blob as 
 * input, whereas the sample x' in the optional third top blob is a sample 
 * from p.
 *
 */
template <typename Dtype>
class VAELayer : public Layer<Dtype> {
 public:
  explicit VAELayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "VAE"; }
  virtual inline int MinNumBottomBlobs() const { return 1; }
  virtual inline int MaxNumBottomBlobs() const { return 2; }
  virtual inline int MinNumTopBlobs() const { return 2; }
  virtual inline int MaxNumTopBlobs() const { return 3; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int n_latent_;
  string encoder_prototxt_, decoder_prototxt_;
  Dtype encoder_loss_weight_;
  bool decoder_only_;

  shared_ptr<Layer<Dtype> > encoder_layer_;
  vector<Blob<Dtype>*> encoder_bottom_vec_;
  vector<Blob<Dtype>*> encoder_top_vec_;

  //so that we can accept gradients coming in from outside via the latent sample
  shared_ptr<Layer<Dtype> > encsample_split_layer_;
  vector<Blob<Dtype>*> encsample_split_bottom_vec_;
  vector<Blob<Dtype>*> encsample_split_top_vec_;


  shared_ptr<Layer<Dtype> > decoder_layer_;
  vector<Blob<Dtype>*> decoder_bottom_vec_;
  vector<Blob<Dtype>*> decoder_top_vec_;

  shared_ptr<Layer<Dtype> > loss_combination_layer_;
  vector<Blob<Dtype>*> loss_combination_bottom_vec_;
  vector<Blob<Dtype>*> loss_combination_top_vec_;

};

}  // namespace caffe

#endif  // CAFFE_VAE_LAYER_HPP_
