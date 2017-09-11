#include <vector>

#include "caffe/layers/two_instance_embed_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void TwoInstanceEmbedDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = NULL;
  if(repeat_inputs_==0 || num_batches_so_far_==0){
    batch=this->prefetch_full_.pop("Data layer prefetch queue empty");
  } else {
    num_batches_so_far_++;
    if(num_batches_so_far_==repeat_inputs_) num_batches_so_far_=0;
  }
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }
  if(top.size()>=3){
    top[2]->ReshapeLike(*(batch->multilabel_[0]));
    caffe_copy(batch->multilabel_[0]->count(),
               batch->multilabel_[0]->gpu_data(),
               top[2]->mutable_gpu_data());
  }

  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  if(repeat_inputs_==0 || num_batches_so_far_==0){
    this->prefetch_free_.push(batch);
  }
}

INSTANTIATE_LAYER_GPU_FORWARD(TwoInstanceEmbedDataLayer);

}  // namespace caffe
