train_id_path = '/content/utils/train_id.py'
val_id_path = '/content/utils/val_id.py'
test_id_path = '/content/utils/test_id.py'
train = 'train'
val = 'val'
test = 'test'

mean_std_map = {
    'train_mean': [0.9211, 0.9211, 0.9211],
    'train_std': [0.1969, 0.1969, 0.1969],
    'val_mean': [0.9228, 0.9228, 0.9228],
    'val_std': [0.1901, 0.1901, 0.1901],
    'test_mean': [0.9407, 0.9407, 0.9407],
    'test_std': [0.1599, 0.1599, 0.1599]
}

# Train Mean: tensor([0.9211, 0.9211, 0.9211])
# Train Std: tensor([0.1969, 0.1969, 0.1969])
# Val Mean: tensor([0.9228, 0.9228, 0.9228]) 
# Val Std: tensor([0.1901, 0.1901, 0.1901])  
# Test Mean: tensor([0.9407, 0.9407, 0.9407])
# Test Std: tensor([0.1599, 0.1599, 0.1599])

