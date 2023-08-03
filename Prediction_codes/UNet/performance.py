# Quantitative analysis for performance analysis
# @author Bereket Kebede, modified from @Dr Cong Van


import numpy as np
import math
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import sys
import os
import natsort
from natsort import natsorted, os_sorted
from tabulate import tabulate
import pandas

# The functions written are based on Luhong Jin 2020
def get_errors(gt,pr, data_range = None):

    # Width and Height of input picture
    W = 256
    H = 256

    # arg = sum( ((gt - pr) ** 2) / (W*H), axis =1)

    arg = ((gt - pr)**2/(W*H)).sum(axis=1)
    arg2 = arg.sum()

    # Peak Signal to Noise ratio
    psnr = 20 * math.log10( (255 / np.sqrt(arg2)) )
    # Standard deviation of the ground truth image
    Sd_gt = gt.std()
    # Normal root mean square error
    nrmse = np.sqrt(arg2)/Sd_gt

    # Calculate PSNR
    #psnr = 20 * math.log10(255 / nrmse * Sd_gt)

    def ssim(gt, pr, data_range = None):
        ssim = structural_similarity(gt, pr, win_size=None, gradient=False, data_range=None, multichannel=True)
        return ssim

    ssim = ssim(gt,pr)
    metrics = [round(nrmse,7), round(ssim,7), round(psnr,7)]
    return metrics

np.set_printoptions(threshold=sys.maxsize)


# control variables
# pred = plt.imread('D:/Bereket/DeepLearning/Analysis/Control variable/Sample_755.tif')
# ground = plt.imread('D:/Bereket/DeepLearning/Analysis/Control variable/Sample_755(2).tif')

# Test_1 - Luhong 2020 Trained, before and after restoration
# Test_3 - CIRL Trained
# Test 4 - CIRL Trained, between ground and predicted
# Test 5 - Luhong 2020 Trained, between ground and predicted

# ground = plt.imread('D:/Bereket/DeepLearning/Testing_codes/UNet/Reproduction/Test_5/Sample_1.tif')
# pred = plt.imread('D:/Bereket/DeepLearning/Testing_codes/UNet/Reproduction/Test_5/Sample_1_pred.tif')

############################################################
# Test 6 - Comparison used for Jin et al paper
# Batch Metrics calculation

# Get prediction and ground images for test presented in the paper[1] supplementary metrics table
# pred_directory_path = 'D:/Bereket/microtubule/Sample/testing_sample_for_paper/Prediction/'
# ground_directory_path = 'D:/Bereket/microtubule/Sample/testing_sample_for_paper/HER/'

# Training data 32-bit
pred_directory_path = 'D:/Bereket/microtubule/Training_Testing_microtubules/Prediction_training_32/'
ground_directory_path = 'D:/Bereket/microtubule/Training_Testing_microtubules/HER/'

# Validation data 16-bit
# pred_directory_path = 'D:/Bereket/microtubule/Training_Testing_microtubules/converted to 16/'
# ground_directory_path = 'D:/Bereket/microtubule/Training_Testing_microtubules/testing_HER/'

# converted ones
# pred_directory_path = 'D:/Bereket/microtubule/Training_Testing_microtubules/converted to 16/'
# ground_directory_path = 'D:/Bereket/microtubule/Training_Testing_microtubules/compare converted to 16/'



Pred_No_of_files = len(os.listdir(pred_directory_path))
Grd_No_of_files = len(os.listdir(ground_directory_path))

# Sort the file names
p = os_sorted(os.listdir(pred_directory_path))
g = os_sorted(os.listdir(ground_directory_path))

all_metric_results = []

for i in range(Pred_No_of_files):
    pred = plt.imread(pred_directory_path+p[i])
    ground = plt.imread(ground_directory_path+g[i])

    #pre_pred = pred
    pre_pred = pred[:, :, 0]
    all_metric_results.append(get_errors(ground, pre_pred))

NRMSE_AVG = 0
SSI_AVG = 0
PSNR_AVG = 0

for i in range (len(all_metric_results)):
    NRMSE_AVG +=all_metric_results[i][0]
    SSI_AVG +=all_metric_results[i][1]
    PSNR_AVG +=all_metric_results[i][2]

#std_all = np.std(all_metric_results)


NRMSE_AVG = NRMSE_AVG/ len(all_metric_results)
SSI_AVG  = SSI_AVG / len(all_metric_results)
PSNR_AVG  = PSNR_AVG / len(all_metric_results)


# print(all_metric_results)
a = [5, 7, 3, 1]
a.sort()


# for i in range (No_of_files):
    # print

# pre_pred = pred[:,:, 0] # use for testing

# pre_pred = pred # use for control variable

#print(pre_pred.shape)

# print(pred.shape)
# print(ground.shape)

# results = get_errors(ground,pre_pred)

# standard deviation for results
std_all = np.std(all_metric_results,axis =0)

print("===========================================================")
print("   Comparing ground truth and prediction quantitatively    ")
print("===========================================================")
print("Data Used: ", 'D:/Bereket/microtubule/Sample/testing_sample_for_paper/')
print("Number of data bundles:", Pred_No_of_files)
print("===========================================================")
print("    NRMSE: |  SSIM: |  PSNR: ")
#print(results)
print(round(NRMSE_AVG,2), '±', round(std_all[0],2), '|' , round(SSI_AVG,2), '±', round(std_all[1],2), '|', round(PSNR_AVG, 2),'±', round(std_all[2],2))
print("===========================================================")
print("Average ± Std_dev")
names = ["NRMSE", "SSIM", "PSNR"]

#print("std dev")

#print(np.std(all_metric_results,axis =0))

#print(pandas.DataFrame((np.std(all_metric_results,axis =0), names)))

#print(np.amax(ground))

