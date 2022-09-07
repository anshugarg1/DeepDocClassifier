from torchvision import transforms as tf
from definitions import mean_std_map

def transformImg(img_h, img_w, mode):
  m = mode+'_mean'
  s = mode+'_std'

  # print(f'{mode} Mean: {mean_std_map[m]}')
  # print(f'{mode} Std: {mean_std_map[s]}')

  trfm = tf.Compose([
                        tf.Resize((img_h,img_w)), 
                        tf.ToTensor(), 
                        tf.Normalize(mean_std_map[m], mean_std_map[s]) 
                    ]) 
  return trfm




#Tobacco-3482 datset
# Train Mean: tensor([0.9211, 0.9211, 0.9211])
# Train Std: tensor([0.1969, 0.1969, 0.1969])
# Val Mean: tensor([0.9228, 0.9228, 0.9228]) 
# Val Std: tensor([0.1901, 0.1901, 0.1901])  
# Test Mean: tensor([0.9407, 0.9407, 0.9407])
# Test Std: tensor([0.1599, 0.1599, 0.1599])

# ImageNet dataset
#mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
