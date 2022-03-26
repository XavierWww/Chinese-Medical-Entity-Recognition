## Chinese Medical Entity Recognition Based on BERT+Bi-LSTM+CRF 

### Step 1

I share the dataset on my google drive, please download the whole 'CCKS_2019_Task1' folder to the current working path.

https://drive.google.com/drive/folders/1Z81nYCnHTvqlzQ0RnO-mFI9xRfJvCf5X?usp=sharing

(Note: There are three empty folds (data, data_test and preprocessed_data) under 'CCKS_2019_Task1' folder)
	
	
### Step 2

Please open 'Preprocess.ipynb' to process raw data.

The processed train data are saved into './CCKS_2019_Task1/data/' by default; 

The processed test data are saved into './CCKS_2019_Task1/data_test/' by default.


### Step 3

Please open 'BERT+Bi_LSTM+CRF.ipynb' to run codes.

You can see I re-process preprocessed data to three '.txt' files for training, validating and testing; 

The three '.txt' files are saved into './CCKS_2019_Task1/processed_data/' by default. And then you can follow my codes to train and test.


### Hope you guys can play my codes successfully and enjoy them !

You also can see a Chinese explanatory article I shared on Zhihu: https://zhuanlan.zhihu.com/p/453350271

If you have any problems, please feel free to contact me via email: [xavier.wu@connect.ust.hk]. 
