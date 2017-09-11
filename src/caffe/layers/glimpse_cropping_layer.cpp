#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/glimpse_cropping_layer.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void GlimpseCroppingLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  GlimpseCroppingParameter glimpse_param = 
       this->layer_param_.glimpse_cropping_param();

  crop_size_=glimpse_param.window();
  do_downsamp_=glimpse_param.do_downsamp();

  CHECK_GT(crop_size_,0) 
      << "Glimpse crop size must be greater than zero";
  CHECK_LT(crop_size_,bottom[0]->shape(2)) 
      << "Glimpse crop size must be less than original image width";
  CHECK_LT(crop_size_,bottom[0]->shape(3)) 
      << "Glimpse crop size must be less than original image height";
}

template <typename Dtype>
void GlimpseCroppingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) 
      << "Input to GlimpseCroppingLayer must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  CHECK_EQ(bottom[1]->shape(1),2) 
      << "Second bottom blob for GlimpseCroppingLayer must be a "
      << "location tuple for each batch element";

  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  if(do_downsamp_){
    top[0]->Reshape(bottom[0]->num(), channels_*2, crop_size_, crop_size_);
  } else {
    top[0]->Reshape(bottom[0]->num(), channels_, crop_size_, crop_size_);
  }

}

template <typename Dtype>
void GlimpseCroppingLayer<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* full_data = bottom[0]->cpu_data();
  const Dtype* location_tuple = bottom[1]->cpu_data();
  const Dtype* embed_bounds = NULL; 
  if(bottom.size()==3){
    embed_bounds=bottom[2]->cpu_data();
  }
  Dtype* glimpse_data = top[0]->mutable_cpu_data();

  for(int iN=0;iN<bottom[0]->shape(0);iN++){
    float crop_x=location_tuple[iN*2]; 
    float crop_y=location_tuple[iN*2+1]; 
    int W=width_;
    int H=height_;
    if(embed_bounds!=NULL){
      H=embed_bounds[2*iN];
      W=embed_bounds[2*iN+1];
      if(W>width_ || H>height_) {
        std::cout << "Oops, looks like you mixed up W and H!" << std::endl;
        exit(0);
      }
    }
    int crop_xmin_pix=(int)(crop_x*(H-crop_size_));
    int crop_ymin_pix=(int)(crop_y*(W-crop_size_));
    for(int ich=0;ich<bottom[0]->shape(1);ich++){
      for(int iX=0;iX<crop_size_;iX++){
        for(int iY=0;iY<crop_size_;iY++){
          glimpse_data[top[0]->offset(iN,ich,iX,iY)]=
            full_data[bottom[0]->offset(iN,ich,crop_xmin_pix+iX,crop_ymin_pix+iY)];
          if(do_downsamp_){
            int crop_xmin_pix_ds=(int)(crop_x*(H-2*crop_size_));
            int crop_ymin_pix_ds=(int)(crop_y*(W-2*crop_size_));

            Dtype downsamp=
              (full_data[bottom[0]->offset(iN,ich,crop_xmin_pix_ds+iX*2,
                                           crop_ymin_pix_ds+iY*2)]
              +full_data[bottom[0]->offset(iN,ich,crop_xmin_pix_ds+iX*2,
                                           crop_ymin_pix_ds+iY*2+1)]
              +full_data[bottom[0]->offset(iN,ich,crop_xmin_pix_ds+iX*2+1,
                                           crop_ymin_pix_ds+iY*2)]
              +full_data[bottom[0]->offset(iN,ich,crop_xmin_pix_ds+iX*2+1,
                                           crop_ymin_pix_ds+iY*2+1)])/4.;
            int topoff=top[0]->offset(iN,ich+bottom[0]->shape(1),iX,iY);
            glimpse_data[topoff]=downsamp;
          }
        }
      }
    }
  }
}

template <typename Dtype>
void GlimpseCroppingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}


#ifdef CPU_ONLY
STUB_GPU(GlimpseCroppingLayer);
#endif

INSTANTIATE_CLASS(GlimpseCroppingLayer);
REGISTER_LAYER_CLASS(GlimpseCropping);

}  // namespace caffe
