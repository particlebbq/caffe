#ifndef CAFFE_UNROLL_LAYER_HPP_
#define CAFFE_UNROLL_LAYER_HPP_

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

 * @brief Given a .prototxt representing a single time slice of a recurrent 
 *        network, constructs an unrolled network out of a specified number 
 *        of time steps of that network.  Bottom blobs of this layer (you can 
 *        have as many as you want) are:
 *          - Initialization blobs.  The prototxt message for this layer 
 *            allows you to specify "recurrent" blobs that transmit data
 *            from one timestep to the next.  If you specify N of these, then 
 *            the first N bottom blobs need to be initializer for these blobs 
 *            at the first time step, in the same order that they are listed 
 *            in the .prototxt.
 *          - Inputs for the time steps.  These input blobs are fed to *all* 
 *            timesteps as input blobs.  So, for example, in a network that 
 *            processes a small patch of a larger image at each timestep, the 
 *            full image would would be passed to each timestep's instance of 
 *            the subnet, and the subnet would be responsible for cropping out 
 *            the patch to be processed at that timestep.
 *        Top blobs of this layer:
 *          - recurrent outputs, i.e. those that transmit information from one 
 *            timestep to the next, such as LSTM hidden/cell layers. These are 
 *            the time-stepped versions of the initialization blobs in the 
 *            bottom list after each timestep, concatenated over time.
 *          - Data products:  any blobs that should be treated as an output of 
 *            the subnet (as listed by the data_product list in the .prototxt);
 *            a blob of the same name is required to be produced by the subnet 
 *            at each timestep.  The blobs produced at each timestep will be
 *            concatenated across time to form a single blob of shape 
 *            (<original shape>,T), which is published as a top blob of this 
 *            layer.
 */
template <typename Dtype>
class UnrollLayer : public Layer<Dtype> {
 public:
  explicit UnrollLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Unroll"; }
  virtual inline int MinBottomBlobs() const { return 0; }
  virtual inline int MaxBottomBlobs() const { return -1; }
  virtual inline int MinNumTopBlobs() const { return 0; }
  virtual inline int MaxNumTopBlobs() const { return -1; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  //the number of forward/backward passes to do over the sampling subnet during 
  //this->Forward()
  int num_timesteps_, N_recur_,N_input_,N_output_;

  string subnet_prototxt_;
  vector<string> recurrent_input_names_,per_time_input_names_,product_names_;
  vector<int> subnet_blob_input_idx_,subnet_blob_output_idx_;

  shared_ptr<Layer<Dtype> > subnet_layer_;

  //indices: [itime][isubnet][iblob]; stores 
  //subnet_layer_subnet()[isubnet]->blobs() at each timestep
  vector<vector<vector<shared_ptr<Blob<Dtype> > > > > subnet_blobs_;
  vector<vector<Blob<Dtype>*> > subnet_bottom_vec_; 
  vector<vector<Blob<Dtype>*> > subnet_top_vec_;   

  //subnet_top_vec_ entries get used by reshape_layer_ as well as by 
  //subnet_bottom_vec_ at the next timestep, so we need a split layer
  vector<vector<shared_ptr<Layer<Dtype> > > > split_layer_;

  //indices: [itime][iprod][iblob(==itime)] where iprod indexes the N_output_ 
  //products of the subnet
  vector<vector<vector<Blob<Dtype>*> > > split_bottom_vec_; 
  vector<vector<vector<Blob<Dtype>*> > > split_top_vec_;

  //consolidate the output blobs across time
  vector<vector<shared_ptr<Layer<Dtype> > > > reshape_layer_;

  //indices: [itime][iprod][iblob(==itime)] where iprod indexes the N_output_ 
  //products of the subnet
  vector<vector<vector<Blob<Dtype>*> > > reshape_bottom_vec_; 
  vector<vector<vector<Blob<Dtype>*> > > reshape_top_vec_;

  vector<shared_ptr<Layer<Dtype> > > concat_layer_;

  //indices: [iprod][iblob(==itime)] where iprod indexes the N_recur_+N_output_ 
  //products of the subnet
  vector<vector<Blob<Dtype>*> > concat_bottom_vec_; 
  vector<vector<Blob<Dtype>*> > concat_top_vec_; 
  
};

}  // namespace caffe

#endif  // CAFFE_UNROLL_LAYER_HPP_
