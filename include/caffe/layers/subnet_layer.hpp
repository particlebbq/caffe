#ifndef CAFFE_SUBNET_LAYER_HPP_
#define CAFFE_SUBNET_MC_LAYER_HPP_

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
 * @brief Encapsulates a sub-network and allows it to interact with the rest of
 *        the network via the standard Layer interface. You can have however 
 *        many bottom and top blobs you want, but the top and bottom blobs 
 *        listed in the prototxt that configure this layer must also exist 
 *        in the .prototxt that defines the sub-network.  If the number of 
 *        bottom blobs supplied to this layer is larger than zero, then the 
 *        first layer defined in the subnet .prototxt should be an InputLayer 
 *        whose top blobs match the list of bottom blobs given to this layer.  
 *        No shapes should be specified in that InputLayer, as these will be 
 *        taken from the bottom blobs given to this layer.
 *
 *        This class is intended to make it easier to work with groups of 
 *        layers that need to be treated differently than the rest of the 
 *        network in some way:  multi-phase training (as in adversarial 
 *        networks), unrolled recurrent networks, etc.
 */
template <typename Dtype>
class SubnetLayer : public Layer<Dtype> {
 public:
  explicit SubnetLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Subnet"; }
  virtual inline int MinNumBottomBlobs() const { return 0; }
  virtual inline int MinNumTopBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  shared_ptr<Net<Dtype> > subnet_;
  int last_layer_index_;
  
  //some applications (like multi-phase training) may want a subnet that only 
  //does, e.g., a backward pass every Nth iteration. This layer supports that 
  //by keeping a counter that tells how many times the subnet has gotten a call
  //to Forward or Backward.  The counter value itself is static because in 
  //general you may want to coordinate this behavior across different subnets 
  //(for example, in a GAN you have a generator and a discriminator, and the 
  //training algorithm has only one getting trained for a while, followed by 
  //an iteration where only the other gets trained).  The next few variables
  //enable that call counting.

  //increments on backward if false; in either case, increment happens only if 
  //phase_counter_index_>=0 and phase_counter_reset_>0
  bool increment_on_forward_; 

  //when phase_counter_[phase_counter_index_]==phase_counter_reset, the phase 
  //counter gets reset to 0 
  int phase_counter_index_, phase_counter_reset_; 
  static vector<int> phase_counter_;

  //meaning of forward_mode_ and backward_mode_:
  //0(default)==>always do forward/back; 
  //1==>only if counter is zero; 
  //2==>only if counter is non-zero; 
  //3==>never
  //also:  to zero out this layer's parameter diffs after the backward pass 
  //has completed use backward_mode==4 (to do it when phase_counter!=0) 
  //or backward_mode==5 (to do it when phase_counter==0)
  int forward_mode_, backward_mode_;  

  vector<bool> shared_inputs_;
  vector<shared_ptr<Blob<Dtype> > > subnet_input_blobs_;
  vector<shared_ptr<Blob<Dtype> > > subnet_output_blobs_;


};


}  // namespace caffe

#endif  // CAFFE_SUBNET_LAYER_HPP_
                                                           
