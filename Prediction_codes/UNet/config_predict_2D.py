##########################################################
# UNet prediction parameters
##########################################################
# parameters
cat_flag = 'test' # Category flag: train, valid, test
Total_epochs = 100  # Number of epochs to predict on
batch_size = 1  # Number of epochs to predict on

shape_of_test_images = 256  # volume dimension of test images (pass in one of the dimension of the plane)
learning_rate = 0.001 # learning rate of the model ot be used
nors_pfp = 3 # number of raw sim per focal plane
##########################################################
# paths
# model to be used for prediction directory

# directory_of_model = 'D:\Bereket\DeepLearning - 3D\Training_codes\UNet\Generated_Model_2D_2\UNet_SIM15_2D_actin_epoch_10_batch_1_lr_0.001.pkl' #%(Total_epochs, batch_size)
# directory_of_model = 'D:/Bereket/DeepLearning - 3D/Training_codes/UNet/Generated Models/Generated_Model_2D_2/UNet_SIM15_2D_actin_epoch_%d_batch_%d_lr_luhong.pkl' %(Total_epochs, batch_size)#  learning_rate)
# directory to put our prediction images
# prediction_out_path = 'D:/Bereket/DeepLearning - 3D/Data/Data_2D_2/%s_prediction/%s_prediction_epoch_%d_batch_%d_lr_luhong'%(cat_flag,cat_flag,Total_epochs, batch_size)
# directory to pull wide-field images from
# to_predict = 'D:/Bereket/DeepLearning - 3D/Data/Data_2D_2/%s'%cat_flag  # path of test images you want to predict
# predict using luhong trained model
# directory_of_model = "D:/Bereket/DeepLearning - 3D/Training_codes/UNet/Generated Models/Luhong_Model/UNet_SIM3_microtubule.pkl"
# prediction_out_path = "D:/Bereket/DeepLearning - 3D/Data/Data_2D_3/test_prediction/test_using_luhong_trained_model"

# Custom paths 1

#to_predict = "D:/Bereket/DeepLearning - 3D/Data/Data_2D_3/test"
#directory_of_model = "D:/Bereket/DeepLearning - 3D/Training_codes/UNet/Generated Models/Generated_Model_2D_3/UNet_SIM%d_2D_actin_epoch_%d_batch_%d_lr_luhong.pkl"%(nors_pfp,Total_epochs, batch_size)
#prediction_out_path = "D:/Bereket/DeepLearning - 3D/Data/Data_2D_3/%s_prediction/%s_prediction_3R_64bit_epoch_%d_batch_%d_lr_luhong"%(cat_flag,cat_flag,Total_epochs, batch_size)

# Custom paths 2

# Predict FairSIM on FairSIM Trained


# Predict FairSIM on Opstad Trained


to_predict = r"E:\Bereket\Research\DeepLearning - 3D\Data\Data_pre_processed\FairSIM\U2OS Actin 2D\Pattern_illuminated"

directory_of_model = r"E:\Bereket\Research\DeepLearning - 3D\Training_codes\UNet\Generated Models\Generated_Model_Data_2D_21\UNet_SIM_3_epoch_2000_batch_8_lr_0.001_luhong.pkl"

prediction_out_path = r"E:\Bereket\Research\DeepLearning - 3D\Data\Data_2D_21\add\config_4\cropped\UNet_SIM_3_epoch_2000_batch_8_lr_0.001_luhong"

prediction_out_full_path = r"E:\Bereket\Research\DeepLearning - 3D\Data\Data_2D_21\add\config_4\full\UNet_SIM_3_epoch_2000_batch_8_lr_0.001_luhong"

# depth of 016 = 9
# depth of 007 = 17
# depth of 007 = 17
# depth of FAIRSIM = 64

my_depth = 64