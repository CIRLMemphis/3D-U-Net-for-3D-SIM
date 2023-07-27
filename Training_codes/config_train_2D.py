
###############################################################
# Training parameters & paths
TOTAL_EPOCHS = 2000
vol_size = 256
batch_size = 8
learning_rate = 0.001 #luhong lr rate
#learning_rate = 0.0001 #ryan lr rate
nors_pfp = 3 #number of planes per focal plane

loss_type = 'luhong'
min_lr = 3e-5
init_lr = 2e-4

# How many epochs until LR s fully decayed to min_lr
decay_steps = TOTAL_EPOCHS//2

# Learning rate gamma to decay to min_lr after decay_steps
lr_gamma = (min_lr/init_lr) ** (1/(decay_steps))

"""Hyperparameters"""
recon_lam = 1.0
#perp_lam = 0.0005
perp_lam = 0.0

child = "Data_2D_21"

train_in_path = "E:/Bereket/Research/DeepLearning - 3D/Data/%s/train"%child
train_gt_path = "E:/Bereket/Research/DeepLearning - 3D/Data/%s/train_gt"%child
valid_in_path = "E:/Bereket/Research/DeepLearning - 3D/Data/%s/valid"%child
valid_gt_path = "E:/Bereket/Research/DeepLearning - 3D/Data/%s/valid_gt"%child


# train_in_path = r"E:\Bereket\Research\DeepLearning - 3D\Data\Data_pre_processed\Opstad\Live Cell\016\2D\Pattern_illuminated"
# train_gt_path = r"E:\Bereket\Research\DeepLearning - 3D\Data\Data_pre_processed\Opstad\Live Cell\016\2D\Reconstructed"
# valid_in_path = r"E:\Bereket\Research\DeepLearning - 3D\Data\Data_pre_processed\Opstad\Live Cell\007\2D\Pattern_illuminated"
# valid_gt_path = r"E:\Bereket\Research\DeepLearning - 3D\Data\Data_pre_processed\Opstad\Live Cell\007\2D\Reconstructed"
###############################################################
# Output paths
model_output_path = "E:/Bereket/Research/DeepLearning - 3D/Training_codes/UNet/Generated Models/Generated_Model_%s"%child
loss_output_path = 'E:/Bereket/Research/DeepLearning - 3D/Training_codes/UNet/Loss function/Loss_%s'%child
log_path = 'E:/Bereket/Research/DeepLearning - 3D/Training_codes/UNet/Generated Models/Generated_Model_%s/logs'%child
