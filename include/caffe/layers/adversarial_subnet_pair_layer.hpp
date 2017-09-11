#ifndef CAFFE_ADVERSARIAL_SUBNET_PAIR_LAYER_HPP_
#define CAFFE_ADVERSARIAL_SUBNET_PAIR_LAYER_HPP_

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
 * @brief Implements the two-phase training that is used for adversarial 
 *        networks, connecting to the rest of the net using the Layer interface.
 *        Keeps an internal counter of the number of calls to Backward.  For 
 *        the first k passes, backpropagation proceeds only through the second 
 *        subnet (normally the discriminator in a typical generative adversarial
 *        model), and only the parameters associated with this subnet are 
 *        updated.  In the next pass, the gradient is propagated through both 
 *        the subnets, and only the first (normally the generator) is updated.
 * 
 *        Note that this implementation has two instances of subnet2 because in
 *        generative adversarial networks, the discriminator (subnet2) takes 
 *        only the generator (subnet1) output when training the generator, but 
 *        takes data as well as gen output when training the discriminator.  
 *        The "f" in subnet2f_ stands for "fixed" (this is the instance that is
 *        used for computing the subnet1 update without modifying the subnet2 
 *        constants.
 */
template <typename Dtype>
class AdversarialSubnetPairLayer : public Layer<Dtype> {
 public:
  explicit AdversarialSubnetPairLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AdversarialSubnetPair"; }
  virtual inline int MinBottomBlobs() const { return 0; }
  virtual inline int MaxBottomBlobs() const { return -1; }
  virtual inline int MinNumTopBlobs() const { return 2; }
  virtual inline int MaxNumTopBlobs() const { return -1; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  //the number of iterations to train subnet2 before spending one on subnet1
  int subnet2_k_;

  //in a generative adversarial network, we need to reverse the direction of 
  //the gradient, so that the generator learns to fool the discriminator rather 
  //than provide more-easily-detectable fakes.  But maybe there's some setup
  //where you don't want to do this?  So it's configurable here.
  bool reverse_subnet1_gradient_;

  //Normally, one wants to do passes over the generated blob and the real data
  //to train the discriminator.  Again, since one may not always 
  //want to do this, here's a switch:
  bool do_external_;

  string subnet1_prototxt_, subnet2_prototxt_;
  int phase_counter_, phase_switch_;

  //the actual layers themselves
  shared_ptr<Layer<Dtype> > subnet1_layer_;
  vector<Blob<Dtype>*> subnet1_bottom_vec_;
  vector<Blob<Dtype>*> subnet1_top_vec_;

  shared_ptr<Layer<Dtype> > subnet2_layer_;
  vector<Blob<Dtype>*> subnet2_bottom_vec_;
  vector<Blob<Dtype>*> subnet2_top_vec_;
  vector<Blob<Dtype>*> subnet2ext_bottom_vec_;
  vector<Blob<Dtype>*> subnet2ext_top_vec_;
  //if do_data_concat_ is true, you probably want to know which patterns are 
  //real data and which are generated.  These blobs provide that info to 
  //subnet2_ and subnet2f_
  Blob<Dtype> is_real_data_,is_not_real_data_;

  //In some applications, subnet1 may require external input (e.g. if the 
  //generator is conditioned on some vector).  The number of such inputs is 
  //N_geninputs_.  N_gen_ is the number of outputs of subnet1, and N_patterns_ 
  //is the size of those output blobs along axis 0.
  int N_geninputs_, N_gen_, N_patterns_;
};

}  // namespace caffe

#endif  // CAFFE_ADVERSARIAL_SUBNET_PAIR_LAYER_HPP_
