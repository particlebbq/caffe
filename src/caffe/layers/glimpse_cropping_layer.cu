#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/glimpse_cropping_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void GlimpseCropForward(const int nthreads,
    const Dtype* const bottom_data, const Dtype* location_tuple, 
    const int num, const int channels, const Dtype* embed_bounds, 
    const int height, const int width, const int crop_size, 
    const bool do_downsamp, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {

    if(index>=nthreads) return;

    const int iN=index/channels;
    const int ich=index%channels;

    int W=width;
    int H=height;
    if(embed_bounds!=NULL){
      H=embed_bounds[2*iN];
      W=embed_bounds[2*iN+1];
    }

    float crop_x=location_tuple[iN*2];  
    float crop_y=location_tuple[iN*2+1];  
    int crop_xmin_pix=(int)(crop_x*(H-crop_size));
    int crop_ymin_pix=(int)(crop_y*(W-crop_size));

    for(int iX=0;iX<crop_size;iX++){
      for(int iY=0;iY<crop_size;iY++){
        if(do_downsamp){
            top_data[(((iN*channels*2)+ich)*crop_size+iX)*crop_size+iY]=
                bottom_data[(((iN*channels)+ich)*height+crop_xmin_pix+iX)
                            *width+crop_ymin_pix+iY];

            int crop_xmin_pix_ds=(int)(crop_x*(H-2*crop_size));
            int crop_ymin_pix_ds=(int)(crop_y*(W-2*crop_size));

            Dtype downsamp=
              (bottom_data[(((iN*channels)+ich)*height+crop_xmin_pix_ds+iX*2)
                           *width+crop_ymin_pix_ds+iY*2]
              +bottom_data[(((iN*channels)+ich)*height+crop_xmin_pix_ds+iX*2+1)
                           *width+crop_ymin_pix_ds+iY*2]
              +bottom_data[(((iN*channels)+ich)*height+crop_xmin_pix_ds+iX*2)
                           *width+crop_ymin_pix_ds+iY*2+1]
              +bottom_data[(((iN*channels)+ich)*height+crop_xmin_pix_ds+iX*2+1)
                           *width+crop_ymin_pix_ds+iY*2+1])/4.;

            int topoff=(((iN*channels*2)+ich+channels)*crop_size+iX)*crop_size+iY;
            top_data[topoff]=downsamp;
        
        } else {
            top_data[(((iN*channels)+ich)*crop_size+iX)*crop_size+iY]=
              bottom_data[(((iN*channels)+ich)*height+crop_xmin_pix+iX)
                          *width+crop_ymin_pix+iY];
        }
      }
    }
  }
}

template <typename Dtype>
void GlimpseCroppingLayer<Dtype>::Forward_gpu(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* full_data = bottom[0]->gpu_data();
  const Dtype* embed_bounds = NULL;
  if(bottom.size()==3){
    embed_bounds=bottom[2]->gpu_data();
  }

  const Dtype* location_tuple = bottom[1]->gpu_data();

  Dtype* glimpse_data = top[0]->mutable_gpu_data();


  GlimpseCropForward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->num()*channels_), CAFFE_CUDA_NUM_THREADS>>>(
      bottom[0]->num()*channels_, full_data, location_tuple, bottom[0]->num(), 
      channels_, embed_bounds,height_, width_, crop_size_, do_downsamp_, 
      glimpse_data);

  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void GlimpseCroppingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}


INSTANTIATE_LAYER_GPU_FUNCS(GlimpseCroppingLayer);


}  // namespace caffe
