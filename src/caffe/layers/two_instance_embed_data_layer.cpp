#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/two_instance_embed_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
TwoInstanceEmbedDataLayer<Dtype>::TwoInstanceEmbedDataLayer(
  const LayerParameter& param) : BasePrefetchingDataLayer<Dtype>(param),
    offset_() {
  db_.reset(db::GetDB(param.data_param().backend()));
  db_->Open(param.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
TwoInstanceEmbedDataLayer<Dtype>::~TwoInstanceEmbedDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void TwoInstanceEmbedDataLayer<Dtype>::DataLayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_batches_so_far_=0;
  repeat_inputs_=this->layer_param_.data_param().repeat_inputs();

  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);

  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  vector<int> prefetch_shape=top_shape;
  top_shape[2] = 100;
  top_shape[3] = 100;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(prefetch_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape;
    label_shape.push_back(batch_size);
    label_shape.push_back(2); //targets
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(label_shape);
    }
  }
  if(top.size()>=3){
    vector<int> location_truth_shape;
    location_truth_shape.push_back(batch_size);
    location_truth_shape.push_back(4);
    top[2]->Reshape(location_truth_shape);
    for(int i=0;i<this->prefetch_.size();i++){
      this->prefetch_[i]->multilabel_.push_back(new Blob<Dtype>());
      this->prefetch_[i]->multilabel_[0]->Reshape(location_truth_shape);
    }
  }
}



template <typename Dtype>
bool TwoInstanceEmbedDataLayer<Dtype>::Skip() {
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();

  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
}

template<typename Dtype>
void TwoInstanceEmbedDataLayer<Dtype>::Next() {
  cursor_->Next();
  if (!cursor_->valid()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Restarting data prefetching from start.";
    cursor_->SeekToFirst();
  }
  offset_++;
}


// This function is called on prefetch thread
template<typename Dtype>
void TwoInstanceEmbedDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  bool do_rand = true;

  Datum datum;
  datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);

  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  top_shape[2] = 100;
  top_shape[3] = 100;

  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  Dtype* loc_label = NULL;
  for(int i=0;i<batch->data_.count();i++) top_data[i]=0;

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  if(batch->multilabel_.size()>0){
    loc_label = batch->multilabel_[0]->mutable_cpu_data();
  }
  Datum datum1;
  Datum datum2;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    while (Skip()) {
      Next();
    }

    if(repeat_inputs_==0 || item_id==0){
      // get a datum
      Next();
      datum1.ParseFromString(cursor_->value());
      datum2.ParseFromString(cursor_->value());
    }
    const string& data1 = datum1.data();
    const string& data2 = datum2.data();

    read_time += timer.MicroSeconds();
    timer.Start();

    int xrange=top_shape[2]-datum1.width();
    int yrange=top_shape[3]-datum1.height();

    int offset1_x,offset1_y,offset2_x,offset2_y;
    Dtype rand[4];
    caffe_rng_uniform<Dtype>(4,0.,1.,rand);
    if(do_rand){
      offset1_x=(int)(xrange*rand[0]);
      offset1_y=(int)(yrange*rand[1]);
      offset2_x=(int)(xrange*rand[2]);
      offset2_y=(int)(yrange*rand[3]);
    } else {
      offset1_x=(int)(xrange*0.25);  
      offset1_y=(int)(yrange*0.5); 
      offset2_x=(int)(xrange*0.75); 
      offset2_y=(int)(yrange*0.5);  
    }


    if(offset1_x>offset2_x) std::swap(offset1_x,offset2_x);
    
    for(int ic=0;ic<datum1.channels();ic++){
      for(int ix=0;ix<datum1.width();ix++){
        for(int iy=0;iy<datum1.height();iy++){
          int top_idx1= ((item_id*datum1.channels()+ic)*top_shape[2]
                         +offset1_y+iy)*top_shape[3]+offset1_x+ix;
          int top_idx2= ((item_id*datum2.channels()+ic)*top_shape[2]
                         +offset2_y+iy)*top_shape[3]+offset2_x+ix;
          int data_index = (ic * datum1.height() + iy) * datum1.width() + ix;  //same for datum1 and datum2
          //scale brightness to [0,1]:  0.00390625=1/256
          Dtype datum1_element = static_cast<Dtype>(
              static_cast<uint8_t>(data1[data_index])) * 0.00390625; 
          Dtype datum2_element = static_cast<Dtype>(
              static_cast<uint8_t>(data2[data_index])) * 0.00390625;
 
          top_data[top_idx1]=datum1_element;
          top_data[top_idx2]=datum2_element;

        }
      }
    }  

    // Copy label.
    if (this->output_labels_) {
      top_label[item_id*2]=datum1.label();
      top_label[item_id*2+1]=datum2.label();
    }
    if(batch->multilabel_.size()>0){
      loc_label[item_id*4]=((Dtype)offset1_x)/xrange;
      loc_label[item_id*4+1]=((Dtype)offset1_y)/yrange;
      loc_label[item_id*4+2]=((Dtype)offset2_x)/xrange;
      loc_label[item_id*4+3]=((Dtype)offset2_y)/yrange;
    }
    trans_time += timer.MicroSeconds();
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}


template <typename Dtype>
void TwoInstanceEmbedDataLayer<Dtype>::Forward_cpu(
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
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
        top[1]->mutable_cpu_data());
  }
  if(top.size()>=3){
    top[2]->ReshapeLike(*(batch->multilabel_[0]));
    caffe_copy(batch->multilabel_[0]->count(),
               batch->multilabel_[0]->cpu_data(),
               top[2]->mutable_cpu_data());
  }
  if(repeat_inputs_==0 || num_batches_so_far_==0){
    this->prefetch_free_.push(batch);
  }
}

INSTANTIATE_CLASS(TwoInstanceEmbedDataLayer);
REGISTER_LAYER_CLASS(TwoInstanceEmbedData);

}  // namespace caffe
