
##############################################
# import libraries
import sys
sys.path.append("C:/Users/CIRL/AppData/Local\Programs/Python/Python39/Lib/site-packages")
sys.path.append(r"E:\Bereket\Research\DeepLearning - 3D\custom_library")

import os
import math
import torch
from torch.utils.data import  DataLoader
from skimage import io, transform
import numpy as np
from unet_model import UNet
import warnings
from tqdm import tqdm
from config_predict import *
import math
from advsr import *
import time

start_time = time.time()

warnings.filterwarnings('ignore')
##############################################
# prevent syntax error
cat_list = ['train','test','valid']

if cat_flag not in cat_list:
    print('wrong cat_flag error. Please use one of the following flags :',*cat_list)
    sys.exit(1)

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
         #print(self.dirs_in[idx])
         train_in_size = 3
         data_in = np.zeros((train_in_size, 64, 64, 64))
         filepath = os.path.join(self.to_predict, self.dirs_in[idx])

         if (train_in_size == 15):
             for i in range(train_in_size):
                 image_name = os.path.join(filepath, "HE_" + str(i + 1) + "." + self.img_type)
                 image = io.imread(image_name)
                 data_in[i, :, :,:] = image

         if (train_in_size == 3):
             for i in range(train_in_size):
                 image_name = os.path.join(filepath, "HE_" + str(5 * i + 1) + "." + self.img_type)
                 #print(image_name)
                 image = io.imread(image_name)
                 data_in[i, :, :,:] = image

         
         #max_in = 5315.0
         #data_in = data_in/1
         sample = {'image_in': data_in,'image_name':self.dirs_in[idx]}
         
         if self.transform:
             sample = self.transform(sample)
        
         return sample

def get_learning_rate(epoch):
    limits = [3, 8, 12]
    lrs = [1, 0.1, 0.05, 0.005]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
        return lrs[-1] * learning_rate

##########################################################################
# Testing starts here

def stitch_prediction_old(input_images):
    # window shape

    auto_grid = int(math.sqrt(len(input_images)))
    grid = auto_grid
    x_grid = grid
    y_grid = grid  # because test data is 1/4 of the dataset
    z_grid = 1

    x, y = [64, 64]
    x, y = [64, 64]
    hor_pixels = 64 * grid

    total = np.zeros([64, hor_pixels, hor_pixels], dtype=np.float32)
    #total = np.zeros([64, hor_pixels, hor_pixels])

    windows = []
    windows_y = []
    count = 0

    for j in range(y_grid):
        for i in range(x_grid):
            # print(count)
            win_x = [i * 64, (i * 64) + 64]
            win_y = [j * 64, (j * 64) + 64]
            windows.append(win_x)
            windows_y.append(win_y)
            total[:, win_y[0]: win_y[1], win_x[0]: win_x[1]] = input_images[count]
            count += 1
    return total

def stitch_prediction(input_images):
    """
    input: Array of cropped imagess
    output: stitched image Z*Y*X
    """
    # if symmetric use auto_grid
    auto_grid = int(np.cbrt(len(input_images)))
    grid = auto_grid
    x_grid = grid
    y_grid = grid
    z_grid = grid

    hor_pixels = 64 * grid
    total = np.zeros([hor_pixels, hor_pixels, hor_pixels], dtype=np.float32)
    count = 0

    for k in range(z_grid):
        for j in range(y_grid):
            for i in range(x_grid):
                # print(count)
                win_x = [i * 64, (i * 64) + 64]
                win_y = [j * 64, (j * 64) + 64]
                win_z = [k * 64, (k * 64) + 64]
                total[win_z[0]: win_z[1], win_y[0]: win_y[1], win_x[0]: win_x[1]] = input_images[count]
                count += 1
    return total

if __name__ == "__main__":
    cuda = torch.device('cuda:0')

    SRRFDATASET = ReconsDataset(to_predict,
                                transform = ToTensor(),
                                img_type = 'tif',
                                in_size = shape_of_test_images)

    test_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size, shuffle=False, pin_memory=True) # better than for loop
    model = UNet(n_channels=Channels, n_classes=1)
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)
    model.load_state_dict(torch.load(directory_of_model))
    model.eval()
    count = 0

    #all_images = np.zeros(64,512,512,64)
    #all_images = np.zeros(64)
    all_images = []


    #for batch_idx, items in tqdm(enumerate(test_dataloader), position=0, desc="idx", leave=False, colour='green', ncols=80):
    for batch_idx, items in enumerate(test_dataloader):

        image = items['image_in']
        image_name = items['image_name']
        count +=1
        sys.stdout.write('\r'+'%d/%d' % (count,len(test_dataloader)))
        model.train()
        #model.eval()
        image = image.float()
        image = image.cuda(cuda)
        pred = model(image)

        # bound output between 0 and 1
        pred[pred < 0] = 0
        pred[pred > 1] = 1

        if not os.path.exists(prediction_out_path):
            os.makedirs(prediction_out_path)

        all_images.append(pred.detach().cpu().numpy())
        #np.append(all_images, pred)
        #np.append(all_images, pred.detach().cpu().numpy())
        #all_images[count-1,:,:] = pred.detach().cpu().numpy()

        io.imsave(prediction_out_path +'/'+image_name[0] + '_pred.tif', pred.detach().cpu().numpy().astype('float32'))
        #io.imsave(prediction_out_path +'/'+image_name[0] + '_pred.tif', pred.detach().cpu().numpy())

    #stitch_prediction(all_images)
    print('Saved to:: ' + prediction_out_path + '/' + image_name[0] + '_pred.tif')

    print("all_images.shape:",len(all_images))

    if not os.path.exists(prediction_out_full_path):
        os.makedirs(prediction_out_full_path)

    # path = r"E:\Bereket\Research\DeepLearning - 3D\Data\Data_3D_17\test_prediction\cropped\config_5"
    # full_path = r"E:\Bereket\Research\DeepLearning - 3D\Data\Data_3D_17\test_prediction\cropped\config_5"

    dirlist = sorted_alphanumeric(os.listdir(prediction_out_path))
    new_Images = []
    for f in dirlist:
        #img = np.squeeze(io.imread(path + '/' + f))
        img = io.imread(prediction_out_path + '/' + f)
        new_Images.append(img)

    #io.imsave(prediction_out_full_path + '/' + 'Stitched' + '_pred_stitched.tif', stitch_prediction(new_Images)[0:my_depth,:,:].astype('float32'))
    io.imsave(prediction_out_full_path + '/' + 'Stitched' + '_pred_stitched.tif', stitch_prediction(new_Images).astype('float32'))

    import subprocess
    subprocess.Popen(r'explorer %s'%prediction_out_full_path)
    print("--- %s seconds ---" % (time.time() - start_time))
