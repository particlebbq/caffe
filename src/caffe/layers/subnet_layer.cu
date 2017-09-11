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
void SubnetLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  bool just_quit(forward_mode_==3);
  if(phase_counter_index_>=0){
    if(forward_mode_==1){
      if(phase_counter_[phase_counter_index_]!=0) just_quit=true;
    } else if(forward_mode_==2){
      if(phase_counter_[phase_counter_index_]==0) just_quit=true;
    }
  }
  if(increment_on_forward_ && 
     phase_counter_index_>=0 && 
     phase_counter_reset_>0) phase_counter_[phase_counter_index_]++;

  if(just_quit) return;

  // Hacky fix for test time, borrowed from RecurrentLayer: reshare all the 
  // internal shared blobs, which may currently point to a stale owner blob 
  // that was dropped when Solver::Test called 
  // test_net->ShareTrainedLayersWith(net_.get()).
  // TODO: somehow make this work non-hackily.

  if (this->phase_ == TEST) {
    subnet_->ShareWeights();
  }

  //Some calling functions (such as UnrollLayer) may want to swap out the 
  //internal blobs used by the subnet.  If this happens, the 
  //subnet_input_blobs_ and subnet_output_blobs_ pointers will be stale.  
  //And since these arrays are not public, we have to fix that here.
  for ( int i = 0; i < bottom.size(); ++i){
    subnet_input_blobs_[i] = 
      subnet_->blob_by_name(this->layer_param_.bottom(i));
    if(!shared_inputs_[i]) continue;
    subnet_input_blobs_[i]->ShareData(*(bottom[i]));
    subnet_input_blobs_[i]->ShareDiff(*(bottom[i]));
  }
  for ( int i = 0; i < top.size(); i++){
    subnet_output_blobs_[i]=subnet_->blob_by_name(this->layer_param_.top(i));
  }

  //can't always rely on solver to have initialized top/bottom diffs to zero 
  //(since there are some layers that do fancy things using this class) so set 
  //them all to zero here:
  for(int i=0;i<subnet_->blobs().size();i++){
    //...but skip input blobs because this will allocate a useless copy of the 
    //data on the gpu
    bool is_input(false);
    for(int j=0;j<subnet_->input_blob_indices().size();j++){
      if(i==subnet_->input_blob_indices()[j]) is_input=true;
    }
    if(is_input && shared_inputs_[i]) continue;

    caffe_gpu_set<Dtype>(subnet_->blobs()[i]->count(),
                         Dtype(0.),
                         subnet_->blobs()[i]->mutable_gpu_diff());
  }

  //fill subnet input blobs with bottom data
  for(int i=0;i<bottom.size();i++){
    if(shared_inputs_[i]){
      subnet_input_blobs_[i]->ShareData(*(bottom[i]));
      subnet_input_blobs_[i]->ShareDiff(*(bottom[i]));
    } else {
      caffe_copy(bottom[i]->count(),
                 bottom[i]->gpu_data(),
                 subnet_input_blobs_[i]->mutable_gpu_data());
    }
  }

  subnet_->Forward();

  //read out top data from subnet and copy it into the top blobs
  for(int i=0;i<top.size();i++){
    caffe_copy(top[i]->count(),
               subnet_output_blobs_[i]->gpu_data(),
               top[i]->mutable_gpu_data());
  }

}

template <typename Dtype>
void SubnetLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
  //internal blobs used by the subnet. If this happens, the 
  //subnet_input_blobs_ and subnet_output_blobs_ pointers will be stale.  And 
  //since these arrays are not public, we have to fix that here.
  for ( int i = 0; i < bottom.size(); ++i){
    subnet_input_blobs_[i] = 
      subnet_->blob_by_name(this->layer_param_.bottom(i));
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
               top[i]->gpu_diff(),
               subnet_output_blobs_[i]->mutable_gpu_diff());
    caffe_copy(top[i]->count(),
               top[i]->gpu_data(),
               subnet_output_blobs_[i]->mutable_gpu_data());
  }

  subnet_->BackwardFrom(last_layer_index_);


  //copy diffs to bottom
  for(int i=0;i<bottom.size();i++){
    if(shared_inputs_[i]) continue;
    caffe_copy(bottom[i]->count(),
               subnet_input_blobs_[i]->gpu_diff(),
               bottom[i]->mutable_gpu_diff());
    caffe_copy(bottom[i]->count(),
               subnet_input_blobs_[i]->gpu_data(),
               bottom[i]->mutable_gpu_data());

  }

  if(phase_counter_index_>=0){
    if((phase_counter_[phase_counter_index_]>0 && backward_mode_==4) || 
       (phase_counter_[phase_counter_index_]==0 && backward_mode_==5)){
      for(int i=0;i<this->blobs_.size();i++){
        caffe_gpu_set<Dtype>(this->blobs_[i]->count(),
                             Dtype(0.),
                             this->blobs_[i]->mutable_gpu_diff());
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SubnetLayer);

}  // namespace caffe
