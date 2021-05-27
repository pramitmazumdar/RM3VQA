# RM3VQA
ICME Grand Challenge 2021 on Quality Assessment of Compressed UGC Videos. Model name RM3VQA submitted by University of Roma Tre, Italy.

# Prerequisites:
Operating system: LINUX/Windows 10

Programming language: MATLAB

Hardware requirements: GPU

# Dependencies:
Packages required: 
1. Deep learning Toolbox (Matlab->Add-ons-> Search Deep learning Toolbox?->Install)
2. Deep learning toolbox for Inception-V3 Network (Matlab->Add-ons-> Search Deep learning Toolbox for Inception-V3" ->Install)
3. Parallel Computing toolbox to perform parallel computation on multicore, and GPU computer. (Matlab -> Add-ons -> Parallel Computing Toolbox-> Install)

# Setup Procedure: 
1. Make RM3VQA as the Root folder.
2. Main_final1.m will be the main program to execute the code.
3. Dataset folder should contain all distorted videos (training/validation/testing) and corresponding Excel file with their subjective scores (MOS/DMOS). 
4. The dataset used for this model consisted of 6300 training videos and 800 testing videos (mp4 format). The video_name_final.csv and videoname_mos_final.csv files consist of the training video names and MOS scores. The training and testing videos are not included in this repository. The test video names and their corresponding MOS should be appended after the training video names/MOS scores in the above two files (preferrably from 6302 row). Although this setting is specifically for the Grand challenge dataset and would change for other dataset.
5. File naming convention in the csv: All filename should be within double inverted comma and should not have file extension. 

# Pretrained models: 
1. Download the pretrained Inception V3 model from https://drive.google.com/file/d/1NfavkJzSnmEW1csQ0vB2dRwmtl600kgH/view?usp=sharing
2. Download the Training Video Features from https://drive.google.com/file/d/1QojwlCj6fzEOojT3ST-w_wZz-UyXzIH1/view?usp=sharing
3. Put these downloaded matlab files in the root directory (main)


# Execution Procedure:
Open Main_final1.m file in Matlab. Change Path to Video and MOS scores in Line 3 and Line 4. Run the Code.

# Output from RM3VQA:
1. Output variable Ypredcknn will be the predicted score for the test dataset

2. SROCC, PLCC, KROCC and RMSE will be the predicted results.

# Dataset used
ICME GC 2021 Dataset (citation to be updated)

# Citation
If you use this model then please cite our work as;

K. Lamichhane, P. Mazumdar, F. Battisti, M. Carli, "A No Reference Deep Learning based Model for Quality Assessment of UGC Videos", International Conference on Multimedia Expo, 2021. (Accepted)


	

