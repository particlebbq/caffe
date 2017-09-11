#include <vector>

#include "caffe/layers/reward_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RewardLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //We need to have:
  // - bottom[0], the predictions blob, with shape (N*M,C,T)
  // - bottom[1], the labels blob, with shape(N,t)
  // - bottom[2], the rewards blob, with shape (N*M,T)
  // - bottom[3] (optional), the curriculum cutoff blob; same shape as bottom[2]
  //Relationships we check here:
  //  - N*M/N = integer
  //  - N*M matches in blobs 0 and 2
  //  - T/t = integer
  //  - T is the same in blobs 0 and 2

  per_target_rewards_=this->layer_param_.reward_param().per_target_rewards();

  int t=1;
  if(bottom[1]->shape().size()>1) t=bottom[1]->shape(1);

  CHECK_EQ(bottom[0]->shape(0)%bottom[1]->shape(0),0)
      << "Prediction and label blobs must have evenly divisible "
      << "numbers of entries on axis 0";
  CHECK_EQ(bottom[0]->shape(0), bottom[2]->shape(0))
      << "Reward must have the same number of entries on axis 0 as "
      << "the predictions do";
  CHECK_EQ(bottom[0]->shape(2)%t,0) 
      << "Prediction and label blob shapes must be (N*M,C,T) and (N,t), "
      << "such that T is evenly divisible by t";

  if(bottom.size()>3) {
    CHECK_EQ(bottom[2]->count(),bottom[3]->count()) 
       << "reward and curriculum cutoff blobs must be the same size";
  }

  CHECK_EQ(bottom[0]->shape(2),bottom[2]->shape(2)) 
      << "Reward blob must have the same number of timesteps as the "
      << "predictions blob";

  num_targets_=bottom[0]->shape(2)/t;

  diff_.ReshapeLike(*(bottom[2]));
  vector<int> top_shape;
  top_shape.push_back(1);
  top[0]->Reshape(top_shape);
  top[1]->ReshapeLike(*(bottom[2]));

}

template <typename Dtype>
void RewardLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int N=bottom[1]->shape(0);
  int M=bottom[0]->shape(0)/N;
  int C=bottom[0]->shape(1);
  int T=bottom[0]->shape(2);
  int t=1;
  if(bottom[1]->shape().size()>1) t=bottom[1]->shape(1);
  int timesteps_per_target=T/t;  
  Dtype* diff=diff_.mutable_cpu_data();     //shape (N*M,T)
  Dtype* curr=diff_.mutable_cpu_diff();     
  const Dtype* pred=bottom[0]->cpu_data();  //shape (N*M,C,T)

  const Dtype* label=bottom[1]->cpu_data(); //shape (N,t)
  Dtype total(0.), total_correct(0.);
  for(int i=0;i<N;i++){
    for(int iM=0;iM<M;iM++){
      bool curriculum_ok(true);
      for(int itarg=0;itarg<t;itarg++){
        for(int istep=0;istep<timesteps_per_target;istep++){
          int idx=iM*N+i;   
          int idx_time=itarg*timesteps_per_target+istep;

          if(per_target_rewards_){
            idx_time=itarg*timesteps_per_target+(timesteps_per_target-1);
          }

          Dtype maxval=-1e30;  
          int maxidx(0);

          for(int j=0;j<C;j++){
            int pred_offset=(idx*C+j)*T+idx_time;

            if(pred[pred_offset]>maxval){
              maxval=pred[pred_offset];
              maxidx=j;
            }
          }
          int label_offset=i*t+itarg;  
          int diff_offset=idx*T+idx_time;
          diff[diff_offset]=0;
          curr[diff_offset]=1;
          if(maxidx==(int)(label[label_offset])){
            diff[diff_offset]=1;
            total_correct+=1;
          } else {
            if(!curriculum_ok) curr[diff_offset]=0;
            if(!per_target_rewards_ || 
               (per_target_rewards_ && istep==timesteps_per_target-1)){
              curriculum_ok=false;
            }
          }
          total+=1.;
        }
      }
    }
  }

  top[0]->mutable_cpu_data()[0] = total_correct/total;
  caffe_copy(diff_.count(),diff_.cpu_data(),top[1]->mutable_cpu_data());
  caffe_copy(diff_.count(),diff_.cpu_data(),top[1]->mutable_cpu_diff());
}

template <typename Dtype>
void RewardLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  int count=bottom[2]->count();
  const Dtype* diff=diff_.cpu_data();
  const Dtype* currvals=diff_.cpu_diff();
  Dtype* reward=bottom[2]->mutable_cpu_diff();
  Dtype* rdat=bottom[2]->mutable_cpu_data();
  Dtype* curr=NULL;
  if(bottom.size()>3) curr=bottom[3]->mutable_cpu_diff();
  for(int i=0;i<count;i++){
    reward[i]=diff[i];
    rdat[i]=diff[i];
    if(curr) curr[i]=currvals[i];
  }
}

#ifdef CPU_ONLY
STUB_GPU(RewardLayer);
#endif

INSTANTIATE_CLASS(RewardLayer);
REGISTER_LAYER_CLASS(Reward);

}  // namespace caffe
