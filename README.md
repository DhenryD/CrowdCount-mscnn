# Installation
1. Install pytorch
2. Clone this repository
  
  We'll call the directory that you cloned CrowdCount-mscnn `ROOT`


# Data Setup
1. Download ShanghaiTech Dataset from  
   
   Baidu Disk: https://pan.baidu.com/s/1FUNQSuezzAQV4e8CDis1yA
   
   Extraction code： 7ous
   
2. Create Directory    ROOT/data/original/shanghaitech/  
 
3. Save "part_A_final" under   ROOT/data/original/shanghaitech/

4. Save "part_B_final" under   ROOT/data/original/shanghaitech/

5. cd ROOT/data_preparation/
   
   run create_gt_test_set_shtech.m in matlab to create ground truth files for test data

6. cd ROOT/data_preparation/
   
   run create_training_set_shtech.m in matlab to create training and validataion set along with ground truth files

# Test
1. Follow steps 1,2,3,4 and 5 from Data Setup

2. Download pre-trained model files(The best model we have trained):

   [[Shanghai Tech A](https://www.dropbox.com/s/8bxwvr4cj4bh5d8/mcnn_shtechA_660.h5?dl=0)]
   
   [[Shanghai Tech B](https://www.dropbox.com/s/kqqkl0exfshsw8v/mcnn_shtechB_110.h5?dl=0)]
   
   Save the model files under ROOT/final_models
   
3. Run test.py

	a. Set save_output = True to save output density maps
	
	b. Errors are saved in  output directory

# Training
1. Follow steps 1,2,3,4 and 6 from Data Setup
2. Run train.py



               

