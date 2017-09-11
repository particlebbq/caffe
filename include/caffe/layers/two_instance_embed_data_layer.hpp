#ifndef CAFFE_TWO_INSTANCE_EMBED_DATA_LAYER_HPP_
#define CAFFE_TWO_INSTANCE_EMBED_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * @brief A data layer for a multi-object detection benchmark, intended for use
 *        with the MNIST handwritten digits dataset.  This class takes two 
 *        instances of the input patterns and randomly places them on a
 *        100x100 blank image.  Produces up to three top blobs:
 *        the 100x100 image itself, a labels blob with shape (N,2) to
 *        specify the two digits, and a location truth blob with 
 *        shape (N,4) to specify the (x,y) coordinates where the two input
 *        instances were placed.
 *       
*/

template <typename Dtype>
class TwoInstanceEmbedDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit TwoInstanceEmbedDataLayer(const LayerParameter& param);
  virtual ~TwoInstanceEmbedDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "TwoInstanceEmbedData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


 protected:
  void Next();
  bool Skip();
  virtual void load_batch(Batch<Dtype>* batch);

  //In reinforcement learning, the update rule involves a sum over MC samples.
  //These samples can be used to produce an improved prediction during the 
  //testing phase.  To that end, if repeat_inputs_ is nonzero, then the all 
  //entries of each group of repeat_inputs_ batches will be populated with the 
  //same input pattern.
  int repeat_inputs_, num_batches_so_far_;
  Batch<Dtype>* last_batch;

  shared_ptr<db::DB> db_;
  shared_ptr<db::Cursor> cursor_;
  uint64_t offset_;

};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
