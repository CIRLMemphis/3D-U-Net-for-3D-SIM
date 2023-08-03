# Predict 2D processed DL-SIM

##########################################################
# Setting up environment
import sys
sys.path.append("C:/Users/CIRL/AppData/Local\Programs/Python/Python39/Lib/site-packages")
sys.path.append("C:/Users/CIRL/AppData/Local\Programs/Python/Python36/Lib/site-packages")
sys.path.append("C:\ProgramData\Miniconda3\lib\site-packages")
sys.path.append(r"E:\Bereket\Research\DeepLearning - 3D\custom_library")

#########################################################
# import libraries
import os
import torch
from torch.utils.data import  DataLoader
from skimage import io, transform
import numpy as np
from unet_model_2D_256 import UNet
import warnings
from config_predict_2D import *
from advsr import *

import time
start_time = time.time()

warnings.filterwarnings('ignore')


#########################################################
# prevent human input error
cat_list = ['train','test','valid']

if cat_flag not in cat_list:
    print('wrong cat_flag error. Please use one of the following flags :',*cat_list)
    sys.exit(1)
#########################################################
# Defining functions
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data_in = sample['image_in']
        name = sample['image_name']
        return {'image_in': torch.from_numpy(data_in),'image_name':name}

class ReconsDataset(torch.utils.data.Dataset):
     def __init__(self, to_predict, transform, img_type,in_size):
        self.to_predict = to_predict
        self.transform = transform
        self.img_type = img_type
        self.in_size = in_size
        self.dirs_in = os.listdir(self.to_predict)
     def __len__(self):
        dirs = os.listdir(self.to_predict)   # open the files
        return len(dirs)            # because one of the file is for groundtruth

     def __getitem__(self, idx): 

         train_in_size = nors_pfp
         data_in = np.zeros((train_in_size, shape_of_test_images, shape_of_test_images))
         filepath = os.path.join(self.to_predict, self.dirs_in[idx])

         # Scrap RAW SIM images
         if (train_in_size == 15):
             for i in range(train_in_size):
                 image_name = os.path.join(filepath, "HE_"+str(i+1)+"." + self.img_type)
                 image = io.imread(image_name)
                 data_in[i,:,:] = image

         if (train_in_size == 3):
             for i in range(train_in_size):
                 image_name = os.path.join(filepath, "HE_"+str(5*i+1)+"." + self.img_type)
                 image = io.imread(image_name)
                 data_in[i,:,:] = image


         #max_in = 5315.0
         #data_in = data_in/np.max(data_in)
         sample = {'image_in': data_in,'image_name':self.dirs_in[idx]}
         
         if self.transform:
             sample = self.transform(sample)
         return sample
########################################################
# Prediction starts here
if __name__ == "__main__":
    cuda = torch.device('cuda:0')

    
    SRRFDATASET = ReconsDataset(to_predict,
                                transform = ToTensor(),
                                img_type = 'tif',
                                in_size = shape_of_test_images)

    test_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=1, shuffle=False, pin_memory=True) # better than for loop
    model = UNet(n_channels=nors_pfp, n_classes=1)
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)
    model.load_state_dict(torch.load(directory_of_model))
    model.eval()
    count = 0

    for batch_idx, items in enumerate(test_dataloader):
        
        image = items['image_in']
        image_name = items['image_name']
        #print(image_name[0])
        #print(r'%d/%d' % (count,len(test_dataloader)) , flush=True)
        sys.stdout.write('\r'+'%d/%d' % (count,len(test_dataloader)))
        count +=1
        #model.eval()
        image = image.float()
        image = image.cuda(cuda)
        pred = model(image)

        pred[pred < 0] = 0
        pred[pred > 1] = 1

        if not os.path.exists(prediction_out_path):
            os.makedirs(prediction_out_path)

        io.imsave(prediction_out_path +'/'+image_name[0] + '_pred.tif', pred.detach().cpu().numpy().astype('float32'))

    print('\n Saved to:: '+ prediction_out_path + '/' + image_name[0] + '_pred.tif')

    ####### stitch images

    if not os.path.exists(prediction_out_full_path):
        os.makedirs(prediction_out_full_path)

    # path = r"E:\Bereket\Research\DeepLearning - 3D\Data\Data_3D_17\test_prediction\cropped\config_5"
    # full_path = r"E:\Bereket\Research\DeepLearning - 3D\Data\Data_3D_17\test_prediction\cropped\config_5"
    dirlist = sorted_alphanumeric(os.listdir(prediction_out_path))
    new_Images = []
    for f in dirlist:
        #img = np.squeeze(io.imread(prediction_out_path + '/' + f))
        img = io.imread(prediction_out_path + '/' + f)
        new_Images.append(img)

    # depth of 016 = 9
    # depth of 007 = 17
    # depth of 007 = 17
    # depth of FAIRSIM = 64
    io.imsave(prediction_out_full_path + '/' + 'pred_stitched.tif', stitch_image_2d_up(new_Images, depth =my_depth).astype('float32'))

    import subprocess
    subprocess.Popen(r'explorer %s' % prediction_out_full_path)
    print("--- %s seconds ---" % (time.time() - start_time))
