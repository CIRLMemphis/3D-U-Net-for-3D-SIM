##########################################################
# UNet prediction parameters

# Category flag: train, valid, test
cat_flag = 'test'
# Number of epochs to predict on
Total_epochs = 100
# Number of epochs to predict on
batch_size = 1
Channels = 3

# frequency of metrics to be tracked, 0.1 means check one every 10 epochs
jump_metrics = 10

# path of test images you want to predict
to_predict = 'E:/Bereket/Research/DeepLearning - 3D/Data/Data_3D_21/%s'%cat_flag
#custom
to_predict = r"E:\Bereket\Research\DeepLearning - 3D\Data\Data_pre_processed\FairSIM\U2OS Actin 3D\Pattern_illuminated"
# volume dimension of test images (pass in one of the dimension of the cube)
shape_of_test_images = 64

learning_rate = 0.0001


# model to be used for prediction directory
# directory_of_model = 'D:/Bereket/DeepLearning - 3D/Training_codes/UNet/Generated_Model_2/UNet_SIM15_3D_microtubule_epoch_%d_batch_%d_lr_0.001.pkl'%(Total_epochs, batch_size)
#directory_of_model = 'D:/Bereket/DeepLearning - 3D/Training_codes/UNet/Generated Models/Generated_Model_3D_3/UNet_SIM15_3D_actin_epoch_%d_batch_%d_lr_0.001.pkl'%(Total_epochs, batch_size)
#directory_of_model = "D:\Bereket\DeepLearning - 3D\Training_codes\UNet\Generated Models\Generated_Model_3D_2\UNet_SIM15_3D_actin_epoch_500_batch_1_lr_luhong.pkl"

# variable link
#directory_of_model = 'E:/Bereket/Research/DeepLearning - 3D/Training_codes/UNet/Generated Models/Generated_Model_3D_9/20210420_H9C2-dTag_GLU_37C_1520_sim-fast_005/UNet_SIM3_3D_20210420_H9C2-dTag_GLU_37C_1520_sim-fast_005_epoch_%d_batch_%d_const_lr_%.4f.pkl'%(Total_epochs, batch_size, learning_rate)

# custom link
#directory_of_model = "E:/Bereket/Research/DeepLearning - 3D/Training_codes/UNet/Generated Models/Generated_Model_3D_7/Generated_Model_3D_7UNet_SIM3_3D_Data_3D_7_epoch_100_batch_1_lr_ryan_0.0001.pkl"

# 016 depth =18
# Actin depth = 64
my_depth = 64

# Predict FairSIM on Opstad Trained

# prediction on Data_3D_24 (Simulated, Star-like) using Data_3D_21


# Predict FairSIM on FairSIM Trained


to_predict = r"E:\Bereket\Research\DeepLearning - 3D\Data\Data_pre_processed\FairSIM\U2OS Actin 3D\Pattern_illuminated"

directory_of_model = r"E:\Bereket\Research\DeepLearning - 3D\Training_codes\UNet\Generated Models\Generated_Model_3D_21\UNet_SIM3_3D_Data_3D_21_epoch_2000_batch_8_lr_ryan_0.0001.pkl"

prediction_out_path = r"E:\Bereket\Research\DeepLearning - 3D\Data\Data_3D_21\add\config_3\cropped\UNet_SIM3_3D_Data_3D_21_epoch_2000_batch_8_lr_ryan_0.0001"

prediction_out_full_path = r"E:\Bereket\Research\DeepLearning - 3D\Data\Data_3D_21\add\config_3\full\UNet_SIM3_3D_Data_3D_21_epoch_2000_batch_8_lr_ryan_0.0001"