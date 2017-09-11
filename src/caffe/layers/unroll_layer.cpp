#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/unroll_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {


template <typename Dtype>
void UnrollLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  num_timesteps_=this->layer_param_.unroll_param().num_timesteps();
  subnet_prototxt_=this->layer_param_.unroll_param().subnet_prototxt();
  N_recur_=this->layer_param_.unroll_param().recurrent_input_size();

  recurrent_input_names_.clear();
  for(int i=0;i<N_recur_;i++) recurrent_input_names_.push_back(
     this->layer_param_.unroll_param().recurrent_input(i));

  //We need to know some things about how many blobs are produced/consumed by 
  //the subnet, so let's take a peek at the prototxt for it before we really 
  //get started setting up layers
  NetParameter net_param_unfilt;  
  ReadNetParamsFromTextFileOrDie(subnet_prototxt_, &net_param_unfilt);
  net_param_unfilt.mutable_state()->set_phase(this->phase_);
  if(this->layer_param_.subnet_param().stage().size()>0){
    net_param_unfilt.mutable_state()->add_stage(
      this->layer_param_.subnet_param().stage());
  }

  NetParameter net_param;
  Net<Dtype>::FilterNet(net_param_unfilt, &net_param);

  const LayerParameter subnet_input_layer_param = net_param.layer(0);
  const InputParameter subnet_input_param = 
      subnet_input_layer_param.input_param();

  std::cout << "unroll layer net param debug string=" 
            << net_param.DebugString() << std::endl;

  //the number of blobs the subnet will take as input at each timestep
  N_input_=0;
  per_time_input_names_.clear();
  if(subnet_input_layer_param.type().compare("Input")==0){
    N_input_=subnet_input_layer_param.top_size()-N_recur_;
    for(int i=0;i<N_input_;i++) per_time_input_names_.push_back(
      subnet_input_layer_param.top(i+N_recur_));
  }

  CHECK_EQ(bottom.size(),N_input_+N_recur_) 
      << "Number of input and recurrent blobs should sum to bottom.size()";

  //the number of blobs the subnet should produce at each timestep
  N_output_=top.size()-N_recur_;
  CHECK_GE(top.size(),N_output_+N_recur_) 
     << "top.size() should be at least as large as N_output_+N_recur_";
  for(int i=N_recur_;i<top.size();i++){
    product_names_.push_back(this->layer_param_.top(i));
  }


  //ok!  now, back to the actual setting up of the things.
  LayerParameter subnet_param;
  subnet_param.set_name(this->layer_param_.name()+"_subnet");
  subnet_param.set_type("Subnet");
  subnet_param.mutable_subnet_param()->set_prototxt_filename(subnet_prototxt_);
  subnet_param.mutable_subnet_param()->set_force_test_phase(
    this->layer_param_.subnet_param().force_test_phase());
  subnet_param.mutable_subnet_param()->set_phase(this->layer_param_.phase());
  subnet_param.set_phase(this->layer_param_.phase());
  if(this->layer_param_.subnet_param().stage().size()>0){
    subnet_param.mutable_subnet_param()->set_stage(
      this->layer_param_.subnet_param().stage());
  }
  subnet_blob_input_idx_.clear(); //coindexed with subnet_bottom_vec_
  vector<string> subnet_blob_input_names;

  split_bottom_vec_.clear();
  split_top_vec_.clear();
  split_layer_.clear();

  split_layer_.resize(num_timesteps_);
  split_bottom_vec_.resize(num_timesteps_);
  split_top_vec_.resize(num_timesteps_);

  for(int itime=0;itime<num_timesteps_;itime++){
    subnet_bottom_vec_.push_back(vector<Blob<Dtype>*>());
    subnet_top_vec_.push_back(vector<Blob<Dtype>*>());

    for(int i=0;i<N_recur_;i++){
      if(itime==0){
        subnet_param.add_bottom(recurrent_input_names_[i]+"_in");
        subnet_param.mutable_subnet_param()->add_shared_inputs(false);
        subnet_bottom_vec_[itime].push_back(bottom[i]);
        subnet_blob_input_names.push_back(recurrent_input_names_[i]+"_in");
      } else {
        subnet_bottom_vec_[itime].push_back(split_top_vec_[itime-1][i][1]);
      }
    } 
    for(int i=0;i<N_input_;i++){
      if(itime==0){
        subnet_param.add_bottom(per_time_input_names_[i]);
        subnet_param.mutable_subnet_param()->add_shared_inputs(true);
        subnet_blob_input_names.push_back(per_time_input_names_[i]);
      }
      subnet_bottom_vec_[itime].push_back(bottom[N_recur_+i]);
      
    }
    for(int i=0;i<N_recur_;i++){
      if(itime==0){
        subnet_param.add_top(recurrent_input_names_[i]+"_out");
      }
      subnet_top_vec_[itime].push_back(new Blob<Dtype>());  
    }
    for(int i=0;i<N_output_;i++){
      if(itime==0) subnet_param.add_top(product_names_[i]);
      subnet_top_vec_[itime].push_back(new Blob<Dtype>());  
    }

    if(itime==0){
      subnet_layer_=LayerRegistry<Dtype>::CreateLayer(subnet_param);
      subnet_layer_->SetUp(subnet_bottom_vec_[0],subnet_top_vec_[0]);
    }
    subnet_layer_->Reshape(subnet_bottom_vec_[itime],subnet_top_vec_[itime]);
    if(itime==0){
      for(int i=0;i<subnet_blob_input_names.size();i++){
        subnet_blob_input_idx_.push_back(
          subnet_layer_->subnets()[0]->blob_names_index().at(
            subnet_blob_input_names[i]));
      }
    }

    for(int i=0;i<N_recur_;i++){
      ostringstream oss;
      oss << itime;
      string itime_str(oss.str());
      LayerParameter split_param;
      split_param.set_name(this->layer_param_.name()
          +"_"+recurrent_input_names_[i]+"_split_"+itime_str);
      split_param.set_type("Split");

      split_bottom_vec_[itime].push_back(vector<Blob<Dtype>*>());
      split_top_vec_[itime].push_back(vector<Blob<Dtype>*>());

      split_param.add_bottom(recurrent_input_names_[i]+"_out");
      split_bottom_vec_[itime][i].push_back(subnet_top_vec_[itime][i]);

      split_param.add_top(recurrent_input_names_[i]+"_split_for_concat_"
          +itime_str);
      split_top_vec_[itime][i].push_back(new Blob<Dtype>());

      split_param.add_top(recurrent_input_names_[i]+"_split_for_next_time_"
          +itime_str);
      split_top_vec_[itime][i].push_back(new Blob<Dtype>());

      split_layer_[itime].push_back(
          LayerRegistry<Dtype>::CreateLayer(split_param));
      split_layer_[itime][i]->SetUp(split_bottom_vec_[itime][i],
          split_top_vec_[itime][i]);
      split_layer_[itime][i]->Reshape(split_bottom_vec_[itime][i],
          split_top_vec_[itime][i]);
    }
  }

  //we want to store the list of intermediate blobs for each timestep in 
  //subnet_blobs_.  Note that blobs() here is Net<Dtype>::blobs(), the list 
  //of intermediates blobs, not the list of learnable parameters.
  subnet_blobs_.resize(num_timesteps_);
  for(int isubnet=0;isubnet<subnet_layer_->subnets().size();isubnet++){
    subnet_blobs_[0].push_back(subnet_layer_->subnets()[isubnet]->blobs());
  }
  for(int itime=1;itime<num_timesteps_;itime++){
    subnet_blobs_[itime].resize(subnet_layer_->subnets().size());
    for(int isubnet=0;isubnet<subnet_layer_->subnets().size();isubnet++){
      subnet_blobs_[itime][isubnet].resize(subnet_blobs_[0][isubnet].size());
      for(int iblob=0;iblob<subnet_blobs_[itime][isubnet].size();iblob++){
        bool found(false);
        if(isubnet==0){
          for(int isearch=0;isearch<subnet_blob_input_idx_.size();isearch++){
            if(iblob==subnet_blob_input_idx_[isearch]) found=true;
          }
        }
        if(found){
          subnet_blobs_[itime][isubnet][iblob]=subnet_blobs_[0][isubnet][iblob];
        } else {
          subnet_blobs_[itime][isubnet][iblob].reset(new Blob<Dtype>());
          subnet_blobs_[itime][isubnet][iblob]->ReshapeLike(
              *(subnet_blobs_[0][isubnet][iblob].get()));
        }
      }
    }
  }


  //Finally, don't forget to concat the subnet products over time
  reshape_bottom_vec_.clear();
  reshape_top_vec_.clear();
  reshape_layer_.clear();

  reshape_layer_.resize(num_timesteps_);
  reshape_bottom_vec_.resize(num_timesteps_);
  reshape_top_vec_.resize(num_timesteps_);

  concat_bottom_vec_.clear();
  concat_top_vec_.clear();
  concat_layer_.clear();
  for(int i=0;i<N_recur_+N_output_;i++){

    for(int itime=0;itime<num_timesteps_;itime++){
      ostringstream oss;
      oss << itime;
      string itime_str(oss.str());
      LayerParameter reshape_param;
      if(i<N_recur_){
        reshape_param.set_name(this->layer_param_.name()+"_"
            +recurrent_input_names_[i]+"_reshape_"+itime_str);
      } else {
        reshape_param.set_name(this->layer_param_.name()+"_"
            +product_names_[i-N_recur_]+"_reshape_"+itime_str);
      }
      reshape_param.set_type("Reshape");

      for(int idim=0;idim<subnet_top_vec_[itime][i]->num_axes();idim++){
        reshape_param.mutable_reshape_param()->mutable_shape()->add_dim(
            subnet_top_vec_[itime][i]->shape(idim));
      }
      reshape_param.mutable_reshape_param()->mutable_shape()->add_dim(1);
      reshape_bottom_vec_[itime].push_back(vector<Blob<Dtype>*>());
      reshape_top_vec_[itime].push_back(vector<Blob<Dtype>*>());

      if(i<N_recur_){
        reshape_param.add_bottom(recurrent_input_names_[i]+"_split_for_concat_"
            +itime_str);
        reshape_bottom_vec_[itime][i].push_back(split_top_vec_[itime][i][0]);
      } else {
        reshape_param.add_bottom(product_names_[i-N_recur_]);
        reshape_bottom_vec_[itime][i].push_back(subnet_top_vec_[itime][i]);
      }
      if(i<N_recur_){
        reshape_param.add_top(recurrent_input_names_[i]+"_reshape_"+itime_str);
      } else {
        reshape_param.add_top(product_names_[i-N_recur_]+"_reshape_"+itime_str);
      }
      reshape_top_vec_[itime][i].push_back(new Blob<Dtype>());
      reshape_layer_[itime].push_back(LayerRegistry<Dtype>::CreateLayer(
          reshape_param));
      reshape_layer_[itime][i]->SetUp(reshape_bottom_vec_[itime][i],
          reshape_top_vec_[itime][i]);
      reshape_layer_[itime][i]->Reshape(reshape_bottom_vec_[itime][i],
          reshape_top_vec_[itime][i]);

    }

    LayerParameter concat_param;
    if(i<N_recur_){
      concat_param.set_name(this->layer_param_.name()+"_"+
          recurrent_input_names_[i]+"_concat");
    } else {
      concat_param.set_name(this->layer_param_.name()+"_"+
          product_names_[i-N_recur_]+"_concat");
    }
    concat_param.set_type("Concat");
    concat_param.mutable_concat_param()->set_axis(
        subnet_top_vec_[0][i]->num_axes()); 
    concat_bottom_vec_.push_back(vector<Blob<Dtype>*>());
    concat_top_vec_.push_back(vector<Blob<Dtype>*>());

    for(int itime=0;itime<num_timesteps_;itime++){
      ostringstream oss;
      oss << itime;
      string itime_str(oss.str());
 
      if(i<N_recur_){
        concat_param.add_bottom(recurrent_input_names_[i]
            +"_reshape_"+itime_str);
      } else {
        concat_param.add_bottom(product_names_[i-N_recur_]
            +"_reshape_"+itime_str);
      }
      concat_bottom_vec_[i].push_back(reshape_top_vec_[itime][i][0]);
    }
    if(i<N_recur_){
      concat_param.add_top(recurrent_input_names_[i]+"_concat");
    } else {
      concat_param.add_top(product_names_[i-N_recur_]+"_concat");
    }
    concat_top_vec_[i].push_back(top[i]);
    concat_layer_.push_back(LayerRegistry<Dtype>::CreateLayer(concat_param));
    concat_layer_[i]->SetUp(concat_bottom_vec_[i],concat_top_vec_[i]);
    concat_layer_[i]->Reshape(concat_bottom_vec_[i],concat_top_vec_[i]);
  }

  

  // This layer's parameters are any parameters in the subnet.
  // Note that blobs() here is Layer<Dtype>::blobs(), the list of learnable 
  //parameters...not the list of intermediate blobs.
  this->blobs_.clear();
  this->blob_names_.clear();
  for (int i = 0; i < subnet_layer_->blobs().size(); ++i) {
    this->blobs_.push_back(subnet_layer_->blobs()[i]);
    this->blob_names_.push_back(subnet_layer_->blob_names()[i]);

    //for every learnable blob, there could in principle be an lr_mult or 
    //decay_mult set
    if(this->layer_param_.param_size()<this->blobs_.size()){
      this->layer_param_.add_param();
    }
    this->layer_param_.mutable_param(this->blobs_.size()-1)->set_lr_mult(
         subnet_layer_->layer_param().param(i).lr_mult());
    this->layer_param_.mutable_param(this->blobs_.size()-1)->set_decay_mult(
         subnet_layer_->layer_param().param(i).decay_mult());;

  }

  //likewise, the intermediates here are the intermediates of the subnet at 
  //each time (i.e. the contents of subnet_blobs_).
  //I think we can safely ignore the reshape and concat top/bottom blobs.
  this->intermediates_.clear();
  this->intermediate_names_.clear();
  for(int itime=0;itime<num_timesteps_;itime++){
    ostringstream oss;
    oss << itime;
    string itime_str(oss.str());
    for(int isubnet=0;isubnet<subnet_blobs_[itime].size();isubnet++){
      ostringstream oss2;
      oss2 << subnet_layer_->subnets()[isubnet]->name();
      string subname(oss2.str());
      for(int iblob=0;iblob<subnet_blobs_[itime][isubnet].size();iblob++){
        this->intermediates_.push_back(subnet_blobs_[itime][isubnet][iblob]);
        this->intermediate_names_.push_back(subname+"::"
           +subnet_layer_->subnets()[isubnet]->blob_names()[iblob]
           +"::time_"+itime_str);
      }
    }
  }

  this->subnets_.clear();
  for(int i=0;i<subnet_layer_->subnets().size();i++){
    this->subnets_.push_back(subnet_layer_->subnets()[i]);
  }

  this->param_propagate_down_.clear();
  this->param_propagate_down_.resize(this->blobs_.size(), true);

}

template <typename Dtype>
void UnrollLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  for(int itime=0;itime<num_timesteps_;itime++){
    subnet_layer_->Reshape(subnet_bottom_vec_[itime],subnet_top_vec_[itime]);
    for(int i=0;i<N_output_+N_recur_;i++){
      reshape_layer_[itime][i]->Reshape(reshape_bottom_vec_[itime][i],
          reshape_top_vec_[itime][i]);
    }
  }
  for(int i=0;i<N_output_+N_recur_;i++){
    concat_layer_[i]->Reshape(concat_bottom_vec_[i],concat_top_vec_[i]);
  }
}

template <typename Dtype>
void UnrollLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  for(int itime=0;itime<num_timesteps_;itime++){
    for(int isubnet=0;isubnet<subnet_blobs_[itime].size();isubnet++){
      subnet_layer_->subnets()[isubnet]->reset_blobs(
          subnet_blobs_[itime][isubnet]);
      for(int iblob=0;iblob<subnet_blobs_[itime][isubnet].size();iblob++) {
        caffe_set(subnet_blobs_[itime][isubnet][iblob]->count(),Dtype(0.),
            subnet_blobs_[itime][isubnet][iblob]->mutable_cpu_diff());
      }
    }

    subnet_layer_->Forward(subnet_bottom_vec_[itime],subnet_top_vec_[itime]);
    for(int i=0;i<N_recur_;i++) {
      split_layer_[itime][i]->Forward(split_bottom_vec_[itime][i],
          split_top_vec_[itime][i]);
    }
    for(int i=0;i<N_output_+N_recur_;i++){
      reshape_layer_[itime][i]->Forward(reshape_bottom_vec_[itime][i],
          reshape_top_vec_[itime][i]);
    }
  }
  for(int i=0;i<N_output_+N_recur_;i++){
    concat_layer_[i]->Forward(concat_bottom_vec_[i],concat_top_vec_[i]);
  }

}

template <typename Dtype>
void UnrollLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  for(int i=0;i<N_output_+N_recur_;i++){
    vector<bool> concat_propagate_down;
    for(int j=0;j<concat_bottom_vec_[i].size();j++){
      concat_propagate_down.push_back(true);  
    }
    concat_layer_[i]->Backward(concat_top_vec_[i],concat_propagate_down,
        concat_bottom_vec_[i]);
  } 

  for(int itime=num_timesteps_-1;itime>=0;itime--){
    for(int i=0;i<N_output_+N_recur_;i++){
      vector<bool> reshape_propagate_down;
      for(int j=0;j<reshape_bottom_vec_[itime][i].size();j++){
        reshape_propagate_down.push_back(true);
      }
      reshape_layer_[itime][i]->Backward(reshape_top_vec_[itime][i],
          reshape_propagate_down,reshape_bottom_vec_[itime][i]);
    }
    for(int i=0;i<N_recur_;i++){
      vector<bool> split_propagate_down;
      for(int j=0;j<split_bottom_vec_[itime][i].size();j++){
        split_propagate_down.push_back(true);
      }
      split_layer_[itime][i]->Backward(split_top_vec_[itime][i],
          split_propagate_down,split_bottom_vec_[itime][i]);

    }

    for(int isubnet=0;isubnet<subnet_blobs_[itime].size();isubnet++){
      subnet_layer_->subnets()[isubnet]->reset_blobs(
          subnet_blobs_[itime][isubnet]);
    }

    vector<bool> subnet_propagate_down;
    for(int i=0;i<subnet_bottom_vec_[itime].size();i++){
      subnet_propagate_down.push_back(true);
    }
    subnet_layer_->Backward(subnet_top_vec_[itime],
      subnet_propagate_down,subnet_bottom_vec_[itime]);

  }

}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(UnrollLayer, Forward);
#endif

INSTANTIATE_CLASS(UnrollLayer);
REGISTER_LAYER_CLASS(Unroll);

}  // namespace caffe
