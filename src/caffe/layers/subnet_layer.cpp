#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/subnet_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template <typename Dtype>
vector<int> SubnetLayer<Dtype>::phase_counter_;

template <typename Dtype>
void SubnetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  string filename = this->layer_param_.subnet_param().prototxt_filename();

  increment_on_forward_=this->layer_param_.subnet_param().increment_on_forward();
  phase_counter_index_=this->layer_param_.subnet_param().phase_counter_index();
  if(phase_counter_.size()<phase_counter_index_+1){
    phase_counter_.resize(phase_counter_index_+1);
  }
  if(phase_counter_index_>=0) phase_counter_[phase_counter_index_]=0;
  phase_counter_reset_=this->layer_param_.subnet_param().phase_counter_reset();
  forward_mode_=this->layer_param_.subnet_param().forward_mode();
  backward_mode_=this->layer_param_.subnet_param().backward_mode();

  shared_inputs_.clear();
  if(this->layer_param_.subnet_param().shared_inputs_size()>0){
    CHECK_EQ(
          this->layer_param_.subnet_param().shared_inputs_size(),
          bottom.size()
    ) << "shared_inputs must be specified once for each bottom blob "
      << "or not at all";
    for(int i=0;i<bottom.size();i++) {
      shared_inputs_.push_back(
         this->layer_param_.subnet_param().shared_inputs(i));
    }
  } else {
    for(int i=0;i<bottom.size();i++) shared_inputs_.push_back(false);
  }

  // Create a NetParameter and bind this layer's bottom/top blobs to its 
  //inputs/outputs.
  NetParameter net_param_unfilt; 
  ReadNetParamsFromTextFileOrDie(filename, &net_param_unfilt);

  if(this->layer_param_.subnet_param().force_test_phase()){
    net_param_unfilt.mutable_state()->set_phase(caffe::TEST);
  } else {
    net_param_unfilt.mutable_state()->set_phase(this->layer_param_.phase());
  }

  if(this->layer_param_.subnet_param().stage().size()>0){
    net_param_unfilt.mutable_state()->add_stage(
      this->layer_param_.subnet_param().stage());
  }

  NetParameter net_param;
  Net<Dtype>::FilterNet(net_param_unfilt, &net_param);
  if(this->layer_param_.subnet_param().force_backward()){
    net_param.set_force_backward(true);
  }

  for(int ilayer=0;ilayer<net_param.layer_size();ilayer++){
    net_param.mutable_layer(ilayer)->set_phase(
        this->layer_param_.subnet_param().phase()); 
    net_param.mutable_layer(ilayer)->mutable_subnet_param()->set_force_test_phase(
        this->layer_param_.subnet_param().force_test_phase());
  }


  LayerParameter* input_layer_param = net_param.mutable_layer(0);
  InputParameter* input_param=NULL;

  subnet_input_blobs_.resize(bottom.size());
  subnet_output_blobs_.resize(top.size());
  if(bottom.size()>0){
    CHECK_EQ(input_layer_param->type().compare("Input"),0) 
        << "First layer of subnet must be input layer";
    input_param = input_layer_param->mutable_input_param();


    CHECK_EQ(input_layer_param->top_size(),bottom.size()) 
        << "Bottom blob list passed to SubnetLayer must match the list "
        << "of input blobs in the subnet";
    for(int i=0;i<input_layer_param->top_size();i++){
      CHECK_EQ(input_layer_param->top(i),this->layer_param_.bottom(i)) 
          << "Each bottom blob name in SubnetLayer must match the "
          << "corresponding input blob in the subnet";
    }

    //configure the subnet's input layer so that it has the correct blob shapes.
    CHECK_EQ(input_param->shape_size(),0) 
        << "Input layer of subnet must not have any blob sizes hard-coded, "
        << "as these are set in SubnetLayer<Dtype>::LayerSetUp";
    for(int iblob=0;iblob<input_layer_param->top_size();iblob++){
      BlobShape input_shape;
      for (int i = 0; i < bottom[iblob]->num_axes(); ++i) {
        input_shape.add_dim(bottom[iblob]->shape(i));
      }
      input_param->add_shape()->CopyFrom(input_shape);
    }
  }

  // Add "pseudo-losses" to all outputs to force backpropagation.
  // (Setting force_backward is too aggressive as we may not need to backprop to
  // all inputs, e.g., the sequence continuation indicators.)
  vector<string> pseudo_losses(this->layer_param_.top_size());
  for (int i = 0; i < this->layer_param_.top_size(); ++i) {
    LayerParameter* layer = net_param.add_layer();
    pseudo_losses[i] = this->layer_param_.top(i) + "_pseudoloss";
    layer->set_name(pseudo_losses[i]);
    layer->set_type("Reduction");
    layer->add_bottom(this->layer_param_.top(i));
    layer->add_top(pseudo_losses[i]);
    layer->add_loss_weight(1);
  }

  //just to make sure there are no name collisions between the subnet and 
  //any other stuff in the network
  const string& layer_name = this->layer_param_.name();
  if (layer_name.size()) {
    for (int i = 0; i < net_param.layer_size(); ++i) {
      LayerParameter* layer = net_param.mutable_layer(i);
      layer->set_name(layer_name + "::" + layer->name());
      layer->set_phase(this->phase_);
      layer->mutable_subnet_param()->set_phase(this->phase_);
    }
  }

  subnet_.reset(new Net<Dtype>(net_param));

  //if this subnet is to be loaded with pretrained constants, now is the time
  if(this->layer_param_.subnet_param().has_pretrained_constants()){
    NetParameter trained_constants;
    ReadNetParamsFromBinaryFileOrDie(
        this->layer_param_.subnet_param().pretrained_constants(),
        &trained_constants);

    if(this->layer_param_.subnet_param().has_strip_pretrained_constants_prefix()){
      string ignore_str=
        this->layer_param_.subnet_param().strip_pretrained_constants_prefix();
      for (int i = 0; i < trained_constants.layer_size(); ++i) {
        LayerParameter* layer = trained_constants.mutable_layer(i);
        if(ignore_str.compare(layer->name())==ignore_str.size()){
          layer->set_name(layer->name().substr(ignore_str.size()));
          layer->set_phase(this->phase_);
          layer->mutable_subnet_param()->set_phase(this->phase_);
        }
      }
    } else {
      for (int i = 0; i < trained_constants.layer_size(); ++i) {
        LayerParameter* layer = trained_constants.mutable_layer(i);
        layer->set_name(layer_name + "::" + layer->name());
        layer->set_phase(this->phase_);
        layer->mutable_subnet_param()->set_phase(this->phase_);
      }
    }
    subnet_->CopyTrainedLayersFrom(trained_constants);
  }


  subnet_->set_debug_info(
      this->layer_param_.subnet_param().debug_info());

  for ( int i = 0; i < bottom.size(); ++i){
    subnet_input_blobs_[i]=subnet_->blob_by_name(this->layer_param_.bottom(i));
    CHECK(subnet_input_blobs_[i]!=NULL)
      << "could not find input blob named " 
      << this->layer_param_.bottom(i) << " in " << filename << std::endl;

    if(shared_inputs_[i]){
      subnet_input_blobs_[i]->ShareData(*(bottom[i]));
      subnet_input_blobs_[i]->ShareDiff(*(bottom[i]));
    }
  }
  for ( int i = 0; i < top.size(); i++){
    subnet_output_blobs_[i]=subnet_->blob_by_name(this->layer_param_.top(i));
    CHECK(subnet_output_blobs_[i]!=NULL)
      << "could not find output blob named " 
      << this->layer_param_.top(i) << " in " << filename << std::endl;
  }

  subnet_->Reshape();

  // This layer's parameters are any parameters in the layers of the subnet.
  // We only want one copy of each parameter, so check that the parameter
  // is "owned" by the layer, rather than shared with another.
  this->blobs_.clear();
  this->blob_names_.clear();
  for (int i = 0; i < subnet_->params().size(); ++i) {
    if (subnet_->param_owners()[i] == -1) {
      LOG(INFO) << "Adding parameter " << i << ": "
                << subnet_->param_display_names()[i];
      LOG(INFO) << this->layer_param_.name() << " learnable blob " 
                << this->blobs_.size() << " is subnet blob " << i;
      this->blobs_.push_back(subnet_->params()[i]);
      this->blob_names_.push_back(subnet_->param_display_names()[i]);

      //for every learnable blob, there could in principle be an lr_mult 
      //or decay_mult set
      if(this->layer_param_.param_size()<this->blobs_.size()) {
        this->layer_param_.add_param();
      }
      this->layer_param_.mutable_param(this->blobs_.size()-1)
          ->set_lr_mult(subnet_->params_lr()[i]);
      this->layer_param_.mutable_param(this->blobs_.size()-1)
          ->set_decay_mult(subnet_->params_weight_decay()[i]);
    }
  }
  
  //intermediate blobs are just the subnet's internal top and bottom blobs
  this->intermediates_.clear();
  this->intermediate_names_.clear();
  for(int i=0;i<subnet_->blobs().size();i++){
    LOG(INFO) << this->layer_param_.name() << " intermediate blob " 
              << this->intermediates_.size() << " is subnet blob " << i 
              << " with name=" << subnet_->blob_names()[i] << std::endl;
    this->intermediates_.push_back(subnet_->blobs()[i]);
    this->intermediate_names_.push_back(subnet_->blob_names()[i]);
  }
  for(int isubnet=0;isubnet<subnet_->subnets().size();isubnet++){
    for(int i=0;i<subnet_->subnets()[isubnet]->blobs().size();i++){
      LOG(INFO) << this->layer_param_.name() << " intermediate blob " 
                << this->intermediates_.size() << " is subnet " << isubnet 
                << " blob " << i << " with name=" 
                << subnet_->subnets()[isubnet]->blob_names()[i] << std::endl;
      this->intermediates_.push_back(subnet_->subnets()[isubnet]->blobs()[i]);
      this->intermediate_names_.push_back(
          subnet_->subnets()[isubnet]->blob_names()[i]);
    }
  }

  //well, ok, it's those plus the intermediates of any layers (of the subnet)
  // that have them
  for(int i=0;i<subnet_->layers().size();i++){
    for(int j=0;j<subnet_->layers()[i]->intermediates().size();j++){
      this->intermediates_.push_back(subnet_->layers()[i]->intermediates()[j]);
      this->intermediate_names_.push_back(
          subnet_->layers()[i]->layer_param().name()
          +"::"+subnet_->layers()[i]->intermediate_names()[j]);
    }
  }




  // Check that param_propagate_down is set for all of the parameters in the
  // subnet; set param_propagate_down to true in this layer.
  for (int i = 0; i < subnet_->layers().size(); ++i) {
    for (int j = 0; j < subnet_->layers()[i]->blobs().size(); ++j) {
      CHECK(subnet_->layers()[i]->param_propagate_down(j) || 
            subnet_->layers()[i]->layer_param().param(j).lr_mult()==0)
          << "param_propagate_down not set for layer " << i << ", named " 
          << subnet_->layers()[i]->layer_param().name() << ", param " << j;
    }
  }
  this->param_propagate_down_.clear();
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  // initialize the diffs of the subnet outputs to 0 -- we will compute 
  // updates in Forward() and Backward()
  for (int i = 0; i < subnet_output_blobs_.size(); ++i) {
    caffe_set(subnet_output_blobs_[i]->count(), Dtype(0),
              subnet_output_blobs_[i]->mutable_cpu_diff());
  }


  // Check that the last layer_param_.top_size() layers are the pseudo-losses;
  // set last_layer_index so that we don't actually run these layers.
  const vector<string>& layer_names = subnet_->layer_names();
  last_layer_index_ = layer_names.size() - 1 - pseudo_losses.size();
  for (int i = last_layer_index_ + 1, j = 0; i < layer_names.size(); ++i, ++j) {
    CHECK_EQ( layer_names[i], layer_name + "::" + pseudo_losses[j]);
  }


  this->subnets_.clear();
  this->subnets_.push_back(subnet_);
  for(int i=0;i<subnet_->subnets().size();i++){
    this->subnets_.push_back(subnet_->subnets()[i]);
  }

  for(int i=0;i<subnet_->layers().size();i++){
    for(int j=0;j<subnet_->layers()[i]->subnets().size();j++){
      this->subnets_.push_back(subnet_->layers()[i]->subnets()[j]);
    }
  }

}

template <typename Dtype>
void SubnetLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  subnet_->Reshape();

  for(int i=0;i<top.size();i++){
    top[i]->ReshapeLike(*(subnet_output_blobs_[i]));
  }

  this->subnets_.clear();
  this->subnets_.push_back(subnet_);

  for(int i=0;i<subnet_->layers().size();i++){
    for(int j=0;j<subnet_->layers()[i]->subnets().size();j++) this->subnets_.push_back(subnet_->layers()[i]->subnets()[j]);
  }

}

template <typename Dtype>
void SubnetLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  bool just_quit(forward_mode_==3);
  if(phase_counter_index_>=0){
    if(forward_mode_==1){
      if(phase_counter_[phase_counter_index_]!=0) just_quit=true;
    } else if(forward_mode_==2){
      if(phase_counter_[phase_counter_index_]==0) just_quit=true;
    }
  }
  if(increment_on_forward_ && phase_counter_index_>=0 && phase_counter_reset_>0) phase_counter_[phase_counter_index_]++;

  if(just_quit) return;

  // Hacky fix for test time, borrowed from RecurrentLayer: reshare all the 
  // internal shared blobs, which may currently point to a stale owner blob 
  // that was dropped when Solver::Test called 
  // test_net->ShareTrainedLayersWith(net_.get()).
  // TODO: somehow make this work non-hackily.

  if (this->phase_ == TEST || true) {
    subnet_->ShareWeights();
  }


  //Some calling functions (such as UnrollLayer) may want to swap out the 
  //internal blobs used by the subnet.  If this happens, the subnet_input_blobs_ 
  //and subnet_output_blobs_ pointers will be stale.  And since these arrays 
  //are not public, we have to fix that here.
  for ( int i = 0; i < bottom.size(); ++i){
    subnet_input_blobs_[i]=subnet_->blob_by_name(this->layer_param_.bottom(i));
    if(!shared_inputs_[i]) continue;
    subnet_input_blobs_[i]->ShareData(*(bottom[i]));
    subnet_input_blobs_[i]->ShareDiff(*(bottom[i]));

  }
  for ( int i = 0; i < top.size(); i++){
    subnet_output_blobs_[i]=subnet_->blob_by_name(this->layer_param_.top(i));
  }


  //can't always rely on solver to have initialized top/bottom diffs to zero 
  //(since there are some layers that do fancy things using this class)
  //so set them all to zero here:
  for(int i=0;i<subnet_->blobs().size();i++){
    caffe_set(subnet_->blobs()[i]->count(),
              Dtype(0.),
              subnet_->blobs()[i]->mutable_cpu_diff());
  }

  //fill subnet input blobs with bottom data
  for(int i=0;i<bottom.size();i++){
    if(!shared_inputs_[i]) continue;
    subnet_input_blobs_[i]->ShareData(*(bottom[i]));
    subnet_input_blobs_[i]->ShareDiff(*(bottom[i]));
  }

  //forward pass for subnet
  subnet_->Forward();

  //read out top data from subnet and copy it into the top blobs
  for(int i=0;i<top.size();i++){
    caffe_copy(top[i]->count(),
               subnet_output_blobs_[i]->cpu_data(),
               top[i]->mutable_cpu_data());
  }

}

template <typename Dtype>
void SubnetLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  bool just_quit(backward_mode_==3);
  if(phase_counter_index_>=0){
    if(backward_mode_==1){
      if(phase_counter_[phase_counter_index_]!=0) just_quit=true;
    } else if(backward_mode_==2){
      if(phase_counter_[phase_counter_index_]==0) just_quit=true;
    }
  }
  if(!increment_on_forward_ && 
      phase_counter_index_>=0 && 
      phase_counter_reset_>0){
    phase_counter_[phase_counter_index_]++;
  }
  if(phase_counter_reset_>0 && phase_counter_index_>=0){
    if(phase_counter_[phase_counter_index_]>phase_counter_reset_){
      phase_counter_[phase_counter_index_]=0;
    }
  }

  if(just_quit) return;


  //Some calling functions (such as UnrollLayer) may want to swap out the 
  //internal blobs used by the subnet.  If this happens, the subnet_input_blobs_ 
  //and subnet_output_blobs_ pointers will be stale.  And since these arrays 
  //are not public, we have to fix that here.
  for ( int i = 0; i < bottom.size(); ++i){
    subnet_input_blobs_[i]=subnet_->blob_by_name(this->layer_param_.bottom(i));
    if(!shared_inputs_[i]) continue;
    subnet_input_blobs_[i]->ShareData(*(bottom[i]));
    subnet_input_blobs_[i]->ShareDiff(*(bottom[i]));

  }
  for ( int i = 0; i < top.size(); i++){
   string blob_name=this->layer_param_.top(i);
    //insert_splits sometimes adds a split for loss functions.  In so doing, it
    //mangles the name of the blob by appending the layer's name.  But I don't 
    //think there's a very easy way to get that layer's name here.  So I'll 
    //just do this hack which will only work if the layer and the blob it 
    //produces have the same name and the blob in question is the first in the 
    //list.
    string split_blob_name=blob_name+"_"+this->layer_param_.name()+"::"+
                           blob_name+"_0_split_0";

    if(subnet_->has_blob(split_blob_name)){
      subnet_output_blobs_[i]=subnet_->blob_by_name(split_blob_name);
    } else {
      subnet_output_blobs_[i]=subnet_->blob_by_name(this->layer_param_.top(i));
    }
  }


  //pass gradients from top blobs to subnet outputs
  for(int i=0;i<top.size();i++){
    caffe_copy(top[i]->count(),
               top[i]->cpu_diff(),
               subnet_output_blobs_[i]->mutable_cpu_diff());
    caffe_copy(top[i]->count(),
               top[i]->cpu_data(),
               subnet_output_blobs_[i]->mutable_cpu_data());
  }

  subnet_->BackwardFrom(last_layer_index_);


  //copy diffs to bottom
  for(int i=0;i<bottom.size();i++){
    if(shared_inputs_[i]) continue;
    caffe_copy(bottom[i]->count(),
               subnet_input_blobs_[i]->cpu_diff(),
               bottom[i]->mutable_cpu_diff());
    caffe_copy(bottom[i]->count(),
               subnet_input_blobs_[i]->cpu_data(),
               bottom[i]->mutable_cpu_data());
  }

  if(phase_counter_index_>=0){
    if((phase_counter_[phase_counter_index_]>0 && backward_mode_==4) || 
       (phase_counter_[phase_counter_index_]==0 && backward_mode_==5)){
      for(int i=0;i<this->blobs_.size();i++){
        caffe_set(this->blobs_[i]->count(),
                  Dtype(0.),
                  this->blobs_[i]->mutable_cpu_diff());
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(SubnetLayer, Forward);
#endif

INSTANTIATE_CLASS(SubnetLayer);
REGISTER_LAYER_CLASS(Subnet);

}  // namespace caffe
