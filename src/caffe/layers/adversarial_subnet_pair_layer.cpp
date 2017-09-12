#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/adversarial_subnet_pair_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template <typename Dtype>
void AdversarialSubnetPairLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  reverse_subnet1_gradient_=
      this->layer_param_.adversarial_pair_param().reverse_subnet1_gradient();
  do_external_=this->layer_param_.adversarial_pair_param().do_data_concat();
  subnet2_k_=this->layer_param_.adversarial_pair_param().subnet2_k();
  subnet1_prototxt_=
      this->layer_param_.adversarial_pair_param().subnet1_prototxt();
  subnet2_prototxt_=
      this->layer_param_.adversarial_pair_param().subnet2_prototxt();


  phase_counter_=0;
  phase_switch_=1;


  //We need to know some things about how many blobs are produced/consumed by 
  //subnet1 and subnet2, so let's take a peek at the prototxt for subnet1 
  //before we really get started setting up layers
  NetParameter net_param_1;  
  ReadNetParamsFromTextFileOrDie(subnet1_prototxt_, &net_param_1);
  const LayerParameter subnet1_input_layer_param = net_param_1.layer(0);
  const InputParameter subnet1_input_param = 
      subnet1_input_layer_param.input_param();

  //the number of (non-random) blobs subnet1 needs as input 
  N_geninputs_=0;
  if(subnet1_input_layer_param.type().compare("Input")==0){
    N_geninputs_=subnet1_input_layer_param.top_size();
  }

  //the number of randomly generated blobs that subnet1 will produce.  
  //Two less than top.size(), because subnet1 outputs are published via the top 
  //list, and the last two top blobs are the subnet1 and subnet2 scores.
  N_gen_=top.size()-2;

  if(do_external_){
    //bottom blobs are the N_geninputs_ vectors that condition subnet1, plus a 
    //real-data example for each of the blobs generated by subnet1
    CHECK_EQ(N_geninputs_+N_gen_,bottom.size()) 
      << "bottom.size() must be equal to top.size()-2 plus the number of blobs "
      << "taken as input by subnet1";
  }

  //the number of patterns in the batch
  N_patterns_=bottom[N_geninputs_]->shape(0);

  //Finally, a quick sanity check: all bottom blobs should have the same size 
  //along axis 0
  for(int i=1;i<bottom.size();i++) {
    CHECK_EQ(bottom[0]->shape(0),bottom[i]->shape(0));
  }


  //ok!  now, back to the actual setting up of the things.
  LayerParameter subnet1_param;
  subnet1_param.set_name(this->layer_param_.name()+"_subnet1");
  subnet1_param.set_type("Subnet");
  subnet1_param.mutable_subnet_param()->set_prototxt_filename(subnet1_prototxt_);
  for(int i=0;i<N_geninputs_;i++){
    subnet1_param.add_bottom(this->layer_param_.bottom(i)); 
    subnet1_bottom_vec_.push_back(bottom[i]);
  } 
  for(int i=0;i<N_gen_;i++){
    subnet1_param.add_top(this->layer_param_.top(i));
    subnet1_top_vec_.push_back(top[i]);
  }
  if(!do_external_){
    subnet1_param.add_top(this->layer_param_.top(N_gen_));  //score for subnet1
    subnet1_top_vec_.push_back(top[N_gen_]);
  }
  subnet1_layer_=LayerRegistry<Dtype>::CreateLayer(subnet1_param);
  subnet1_layer_->SetUp(subnet1_bottom_vec_,subnet1_top_vec_);
  subnet1_layer_->Reshape(subnet1_bottom_vec_,subnet1_top_vec_);

  //We also need to assemble a list of "is_real_data" labels for the input to
  //the discriminator subnet
  vector<int> is_real_data_shape;
  is_real_data_shape.push_back(N_patterns_);
  is_real_data_shape.push_back(1);
  is_real_data_.Reshape(is_real_data_shape);
  is_not_real_data_.Reshape(is_real_data_shape);
  
  //...and since this never changes, we might as well just fill it now
  Dtype* ird=is_real_data_.mutable_cpu_data();
  Dtype* ird2=is_not_real_data_.mutable_cpu_data();
  for(int i=0;i<N_patterns_;i++){
    ird[i]=1;
    ird2[i]=0;
  }


  LayerParameter subnet2_param;
  subnet2_param.set_name(this->layer_param_.name()+"_subnet2");
  subnet2_param.set_type("Subnet");
  subnet2_param.mutable_subnet_param()->set_prototxt_filename(subnet2_prototxt_);
  if(do_external_){
    for(int i=0;i<N_gen_;i++){
      subnet2_param.add_bottom(this->layer_param_.top(i));
      subnet2_bottom_vec_.push_back(subnet1_top_vec_[i]);
      subnet2ext_bottom_vec_.push_back(bottom[N_geninputs_+i]);
    }
    subnet2_param.add_bottom(this->layer_param_.name()+"_is_real_data");
    subnet2_bottom_vec_.push_back(&is_not_real_data_);
    subnet2ext_bottom_vec_.push_back(&is_real_data_);
  } else {
    for(int i=0;i<N_gen_;i++){
      subnet2_param.add_bottom(this->layer_param_.top(i));    
      subnet2_bottom_vec_.push_back(top[i]);
    }
  }
  subnet2_param.add_top("loss"); 
  subnet2_top_vec_.push_back(top[N_gen_]);
  subnet2ext_top_vec_.push_back(top[N_gen_+1]);
  subnet2_layer_=LayerRegistry<Dtype>::CreateLayer(subnet2_param);
  subnet2_layer_->SetUp(subnet2_bottom_vec_,subnet2_top_vec_);
  subnet2_layer_->Reshape(subnet2_bottom_vec_,subnet2_top_vec_);
  subnet2_layer_->Reshape(subnet2ext_bottom_vec_,subnet2ext_top_vec_);

  
  // This layer's parameters are any parameters in subnet1_layer_ 
  // and subnet2_layer_
  this->blobs_.clear();
  for (int i = 0; i < subnet1_layer_->blobs().size(); ++i) {
    this->blobs_.push_back(subnet1_layer_->blobs()[i]);
    this->blob_names_.push_back(
      this->layer_param_.name()+"::subnet1::"+subnet1_layer_->blob_names()[i]);
  }
  for (int i = 0; i < subnet2_layer_->blobs().size(); ++i) {
    this->blobs_.push_back(subnet2_layer_->blobs()[i]);
    this->blob_names_.push_back(
      this->layer_param_.name()+"::subnet2::"+subnet2_layer_->blob_names()[i]);
  }

  //intermediates and subnets
  this->intermediates_.clear();
  this->intermediate_names_.clear();
  for(int i=0;i<subnet1_layer_->intermediates().size();i++){
    this->intermediates_.push_back(subnet1_layer_->intermediates()[i]);
    this->intermediate_names_.push_back(
      subnet1_layer_->intermediate_names()[i]);
  }
  for(int i=0;i<subnet2_layer_->intermediates().size();i++){
    this->intermediates_.push_back(subnet2_layer_->intermediates()[i]);
    this->intermediate_names_.push_back(
      subnet2_layer_->intermediate_names()[i]);
  }


  this->subnets_.clear();
  for(int i=0;i<subnet1_layer_->subnets().size();i++){
    this->subnets_.push_back(subnet1_layer_->subnets()[i]);
  }
  for(int i=0;i<subnet2_layer_->subnets().size();i++){
    this->subnets_.push_back(subnet2_layer_->subnets()[i]);
  }


  this->param_propagate_down_.clear();
  this->param_propagate_down_.resize(this->blobs_.size(), true);

}

template <typename Dtype>
void AdversarialSubnetPairLayer<Dtype>::Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  subnet1_layer_->Reshape(subnet1_bottom_vec_,subnet1_top_vec_);
  subnet2_layer_->Reshape(subnet2_bottom_vec_,subnet2_top_vec_);
  subnet2_layer_->Reshape(subnet2ext_bottom_vec_,subnet2ext_top_vec_);

}

template <typename Dtype>
void AdversarialSubnetPairLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  subnet1_layer_->Forward(subnet1_bottom_vec_,subnet1_top_vec_);
  subnet2_layer_->Forward(subnet2_bottom_vec_,subnet2_top_vec_);
  

  vector<bool> subnet2_propagate_down;
  for(int i=0;i<subnet2_bottom_vec_.size();i++){
    subnet2_propagate_down.push_back(true);
  }
  subnet2_layer_->Backward(subnet2_top_vec_,
                           subnet2_propagate_down,
                           subnet2_bottom_vec_);

  //phase_switch:  
  //  1==> subnet1 is getting trained: run Fwd passes over generated images 
  //       only; discard subnet2 gradients
  //  2==> subnet2 is getting trained: run Fwd passes over both generated and 
  //       external inputs, but no backward pass for subnet1
  if(phase_switch_==1){

    vector<bool> subnet1_propagate_down;
    for(int i=0;i<subnet1_bottom_vec_.size();i++){
      subnet1_propagate_down.push_back(true);
    }
    subnet1_layer_->Backward(subnet1_top_vec_,
                             subnet1_propagate_down,
                             subnet1_bottom_vec_);

    for(int i=0;i<subnet2_layer_->subnets().size();i++){
      subnet2_layer_->subnets()[i]->ClearParamDiffs();
    }
  } else {
    subnet2_layer_->Forward(subnet2ext_bottom_vec_,subnet2ext_top_vec_);
    subnet2_layer_->Backward(subnet2ext_top_vec_,
                             subnet2_propagate_down,
                             subnet2ext_bottom_vec_);  //this should accumulate with the previous call

  }
}

template <typename Dtype>
void AdversarialSubnetPairLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


  //phase_switch:  
  //  1==> subnet1 is getting trained: run Fwd passes over generated images 
  //       only; discard subnet2 gradients
  //  2==> subnet2 is getting trained: run Fwd passes over both generated and 
  //       external inputs, but no backward pass for subnet1
  //...gradients have been computed in Forward, so there's only a little bit to do here.

  if(phase_switch_==1){
    if(reverse_subnet1_gradient_){
      for(int i=0;i<subnet1_layer_->blobs().size();i++) {
        caffe_scal(subnet1_layer_->blobs()[i]->count(),
                   Dtype(-1.),
                   subnet1_layer_->blobs()[i]->mutable_cpu_diff());
      }
    }
  }


  phase_counter_++;
  if(phase_switch_==1) {
    if(phase_counter_>=subnet2_k_) {
      phase_switch_=2;
      phase_counter_=0;
    }
  } else {
    if(phase_counter_>=1){
      phase_switch_=1;
      phase_counter_=0;
    }
  }

}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(AdversarialSubnetPairLayer, Forward);
#endif

INSTANTIATE_CLASS(AdversarialSubnetPairLayer);
REGISTER_LAYER_CLASS(AdversarialSubnetPair);

}  // namespace caffe