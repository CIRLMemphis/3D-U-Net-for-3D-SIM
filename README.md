## Comparison between 2D and 3D U-Net for 3D SIM Super-resolution

<!-- GETTING STARTED -->
## Getting Started

This 3D U-Net code framework is an extension of Luhong Jin's 2D U-Net, it uses 3D processing instead of 2D processing and achieves less artifacts than its corresponding 2D U-Net.
Questions about this repo can be forwarded to bkebede@memphis.edu. This work was done under as part of research in Computational Imaging Research Lab (CIRL) at the University of Memphis.

### Prerequisites

Windows PC core @i7, 32 GB, NVDIA TITAN RTX GPU
* pytorch
  ```sh
  pip install pytorch
  ```

<!-- USAGE EXAMPLES -->
## Usage

Training : run "training_UNet_SIM_3D.py" to train 3D U-Net  <br>
Training config file: 3D U-Net config file stored at "config_train_3D.py"  <br>
Testing: run "Prediction_UNet_SIM_3D.py" to test 3D U-Net <br>
Testing config file: 3D U-Net config file stores at "config_predict.py" <br>

## File Descriptions

Analysis - visualization and metrics comparison of results. <br>
Data - data couldn't be uploaded on github, its on local GPU computer in "E/Hasti"  <br>
Training codes - python code for training 3D U-Net  <br>
Prediction_codes - python code for running restoration inference. <br>
Workspace - code for pre-processing data <br>
custom_library - "advsr.py" (Advancned Super-Resolution library). a supplementary library functions for 3D UNet. <be>

