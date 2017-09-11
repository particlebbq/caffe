#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/vae_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template <typename Dtype>
void VAELayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  encoder_prototxt_=
      this->layer_param_.vae_param().encoder_prototxt();
  decoder_prototxt_=
      this->layer_param_.vae_param().decoder_prototxt();
  encoder_loss_weight_=
      this->layer_param_.vae_param().encoder_loss_weight();
  decoder_only_=
      this->layer_param_.vae_param().decoder_only();

  if(decoder_only_){


    LayerParameter encoder_param;
    encoder_param.set_name(this->layer_param_.name()+"_encoder");
    encoder_param.set_type("Subnet");
    encoder_param.mutable_subnet_param()->set_prototxt_filename(encoder_prototxt_);
    encoder_param.add_bottom("X");  
    encoder_bottom_vec_.push_back(bottom[1]);
    encoder_param.add_top("loss");
    encoder_top_vec_.push_back(new Blob<Dtype>());
    encoder_param.add_top("sample");
    encoder_top_vec_.push_back(top[1]);
    encoder_layer_=LayerRegistry<Dtype>::CreateLayer(encoder_param);
    encoder_layer_->SetUp(encoder_bottom_vec_,encoder_top_vec_);
    encoder_layer_->Reshape(encoder_bottom_vec_,encoder_top_vec_);


    LayerParameter decoder_param;
    decoder_param.set_name(this->layer_param_.name()+"_decoder");
    decoder_param.set_type("Subnet");
    decoder_param.mutable_subnet_param()->set_prototxt_filename(decoder_prototxt_);
    decoder_param.add_bottom("latent");
    decoder_bottom_vec_.push_back(bottom[0]);
    decoder_param.add_bottom("X");
    decoder_bottom_vec_.push_back(bottom[1]);
    decoder_param.add_top("loss");
    decoder_top_vec_.push_back(new Blob<Dtype>());
    if(top.size()>=3){
      decoder_param.add_top("sample");
      decoder_top_vec_.push_back(top[2]);
    }
    decoder_layer_=LayerRegistry<Dtype>::CreateLayer(decoder_param);
    decoder_layer_->SetUp(decoder_bottom_vec_,decoder_top_vec_);
    decoder_layer_->Reshape(decoder_bottom_vec_,decoder_top_vec_);


    LayerParameter loss_combination_param;
    loss_combination_param.set_name(this->layer_param_.name()+"_loss_combination");
    loss_combination_param.set_type("Eltwise");
    loss_combination_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
    loss_combination_param.mutable_eltwise_param()->add_coeff(encoder_loss_weight_);
    loss_combination_param.mutable_eltwise_param()->add_coeff(1);
    loss_combination_param.add_bottom("encoder_loss");
    loss_combination_bottom_vec_.push_back(encoder_top_vec_[0]);
    loss_combination_param.add_bottom("decoder_loss");
    loss_combination_bottom_vec_.push_back(decoder_top_vec_[0]);
    loss_combination_param.add_top(this->layer_param_.top(0));
    loss_combination_top_vec_.push_back(top[0]);
    loss_combination_layer_=
        LayerRegistry<Dtype>::CreateLayer(loss_combination_param);
    loss_combination_layer_->SetUp(loss_combination_bottom_vec_,
                                   loss_combination_top_vec_);
    loss_combination_layer_->Reshape(loss_combination_bottom_vec_,
                                    loss_combination_top_vec_);

  } else {

    LayerParameter encoder_param;
    encoder_param.set_name(this->layer_param_.name()+"_encoder");
    encoder_param.set_type("Subnet");
    encoder_param.mutable_subnet_param()->set_prototxt_filename(encoder_prototxt_);
    encoder_param.add_bottom("X"); 
    encoder_bottom_vec_.push_back(bottom[0]);
    encoder_param.add_top("loss");
    encoder_top_vec_.push_back(new Blob<Dtype>());
    encoder_param.add_top("sample");
    encoder_top_vec_.push_back(new Blob<Dtype>()); 
    encoder_layer_=LayerRegistry<Dtype>::CreateLayer(encoder_param);
    encoder_layer_->SetUp(encoder_bottom_vec_,encoder_top_vec_);
    encoder_layer_->Reshape(encoder_bottom_vec_,encoder_top_vec_);

    LayerParameter encsample_split_param;
    encsample_split_param.set_name(this->layer_param_.name()+"_encsample_split");
    encsample_split_param.set_type("Split");
    encsample_split_param.add_bottom("enc_sample");
    encsample_split_bottom_vec_.push_back(encoder_top_vec_[1]);
    encsample_split_param.add_top("enc_sample_top");
    encsample_split_top_vec_.push_back(top[1]);
    encsample_split_param.add_top("enc_sample_decoder");
    encsample_split_top_vec_.push_back(new Blob<Dtype>());
    encsample_split_layer_=LayerRegistry<Dtype>::CreateLayer(encsample_split_param);
    encsample_split_layer_->SetUp(encsample_split_bottom_vec_,encsample_split_top_vec_);
    encsample_split_layer_->Reshape(encsample_split_bottom_vec_,encsample_split_top_vec_);


    LayerParameter decoder_param;
    decoder_param.set_name(this->layer_param_.name()+"_decoder");
    decoder_param.set_type("Subnet");
    decoder_param.mutable_subnet_param()->set_prototxt_filename(decoder_prototxt_);
    decoder_param.add_bottom("latent");    
    decoder_bottom_vec_.push_back(encsample_split_top_vec_[1]);
    decoder_param.add_bottom("X");
    decoder_bottom_vec_.push_back(bottom[0]);
    decoder_param.add_top("loss");
    decoder_top_vec_.push_back(new Blob<Dtype>());
    if(top.size()>=3){
      decoder_param.add_top("sample");
      decoder_top_vec_.push_back(top[2]);
    }
    decoder_layer_=LayerRegistry<Dtype>::CreateLayer(decoder_param);
    decoder_layer_->SetUp(decoder_bottom_vec_,decoder_top_vec_);
    decoder_layer_->Reshape(decoder_bottom_vec_,decoder_top_vec_);
  
    LayerParameter loss_combination_param;
    loss_combination_param.set_name(this->layer_param_.name()+"_loss_combination");
    loss_combination_param.set_type("Eltwise");
    loss_combination_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
    loss_combination_param.mutable_eltwise_param()->add_coeff(encoder_loss_weight_); 
    loss_combination_param.mutable_eltwise_param()->add_coeff(1);  
    loss_combination_param.add_bottom("encoder_loss");
    loss_combination_bottom_vec_.push_back(encoder_top_vec_[0]);
    loss_combination_param.add_bottom("decoder_loss");
    loss_combination_bottom_vec_.push_back(decoder_top_vec_[0]);
    loss_combination_param.add_top(this->layer_param_.top(0));
    loss_combination_top_vec_.push_back(top[0]);
    loss_combination_layer_=
        LayerRegistry<Dtype>::CreateLayer(loss_combination_param);
    loss_combination_layer_->SetUp(loss_combination_bottom_vec_,
                                   loss_combination_top_vec_);
    loss_combination_layer_->Reshape(loss_combination_bottom_vec_,
                                    loss_combination_top_vec_);
  
  }


  // This layer's parameters are any parameters in encoder_layer_ 
  // and decoder_layer_
  this->blobs_.clear();
  this->blob_names_.clear();
  for (int i = 0; i < encoder_layer_->blobs().size(); ++i) {
    this->blobs_.push_back(encoder_layer_->blobs()[i]);
    this->blob_names_.push_back(this->layer_param_.name()
                               +"::encoder::"+encoder_layer_->blob_names()[i]);
  }
  for (int i = 0; i < decoder_layer_->blobs().size(); ++i) {
    this->blobs_.push_back(decoder_layer_->blobs()[i]);
    this->blob_names_.push_back(this->layer_param_.name()
                               +"::decoder::"+decoder_layer_->blob_names()[i]);
  }


  this->intermediates_.clear();
  this->intermediate_names_.clear();
  for(int i=0;i<encoder_layer_->intermediates().size();i++){
    this->intermediates_.push_back(encoder_layer_->intermediates()[i]);
    this->intermediate_names_.push_back(encoder_layer_->intermediate_names()[i]);
  }
  for(int i=0;i<decoder_layer_->intermediates().size();i++){
    this->intermediates_.push_back(decoder_layer_->intermediates()[i]);
    this->intermediate_names_.push_back(decoder_layer_->intermediate_names()[i]);
  }


  this->subnets_.clear();
  for(int i=0;i<encoder_layer_->subnets().size();i++){
    this->subnets_.push_back(encoder_layer_->subnets()[i]);
  }
  for(int i=0;i<decoder_layer_->subnets().size();i++){
    this->subnets_.push_back(decoder_layer_->subnets()[i]);
  }


  this->param_propagate_down_.clear();
  this->param_propagate_down_.resize(this->blobs_.size(), true);

}

template <typename Dtype>
void VAELayer<Dtype>::Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  if(decoder_only_){
    encoder_layer_->Reshape(encoder_bottom_vec_,encoder_top_vec_);
    decoder_layer_->Reshape(decoder_bottom_vec_,decoder_top_vec_);
  } else {
    encoder_layer_->Reshape(encoder_bottom_vec_,encoder_top_vec_);
    encsample_split_layer_->Reshape(encsample_split_bottom_vec_,encsample_split_top_vec_);
    decoder_layer_->Reshape(decoder_bottom_vec_,decoder_top_vec_);
    loss_combination_layer_->Reshape(loss_combination_bottom_vec_,loss_combination_top_vec_);
  }
}  

template <typename Dtype>
void VAELayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  if(decoder_only_){
    decoder_layer_->Forward(decoder_bottom_vec_,decoder_top_vec_);
  } else {
    encoder_layer_->Forward(encoder_bottom_vec_,encoder_top_vec_);
    encsample_split_layer_->Forward(encsample_split_bottom_vec_,encsample_split_top_vec_);
    decoder_layer_->Forward(decoder_bottom_vec_,decoder_top_vec_);
    loss_combination_layer_->Forward(loss_combination_bottom_vec_,loss_combination_top_vec_);
  }
}

template <typename Dtype>
void VAELayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if(decoder_only_) CHECK(false) << "No backward in decoder-only mode";

  vector<bool> loss_combination_propagate_down;
  loss_combination_propagate_down.push_back(true);
  loss_combination_propagate_down.push_back(true);
  loss_combination_layer_->Backward(loss_combination_top_vec_,
                                    loss_combination_propagate_down,
                                    loss_combination_bottom_vec_);

   vector<bool> decoder_propagate_down;
  for(int i=0;i<decoder_bottom_vec_.size();i++){
    decoder_propagate_down.push_back(true);
  }
  decoder_layer_->Backward(decoder_top_vec_,
                           decoder_propagate_down,
                           decoder_bottom_vec_);


  vector<bool> encsample_split_propagate_down;
  for(int i=0;i<encsample_split_bottom_vec_.size();i++){
    encsample_split_propagate_down.push_back(true);
  }
  encsample_split_layer_->Backward(encsample_split_top_vec_,
                                   encsample_split_propagate_down,
                                   encsample_split_bottom_vec_);

  vector<bool> encoder_propagate_down;
  for(int i=0;i<encoder_bottom_vec_.size();i++){
    encoder_propagate_down.push_back(true);
  }
  encoder_layer_->Backward(encoder_top_vec_,
                           encoder_propagate_down, 
                           encoder_bottom_vec_);

}

INSTANTIATE_CLASS(VAELayer);
REGISTER_LAYER_CLASS(VAE);

}  // namespace caffe
