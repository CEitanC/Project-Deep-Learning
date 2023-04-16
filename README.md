# Deep Learning – 46211
## Project - DL for tabular data
![image](https://user-images.githubusercontent.com/94564657/215352447-d79c0427-c497-4e0b-863a-7861683c67e8.png)

---

## Project Content
- Dataset - COVID-19 patient's symptoms, status, and medical history
- Data Pre-Processing and Sampling Techniques
- XGBoost and simple FC neural network
- TabNet - deep learning architecture designed by Google to affectively apply deep learning on tabular data
- Additional method for improving results
- Results Comparison

---

## Repository Content
- Two helpers files:
  1. models_helpers.py
  2. pre_proccessing_helpers.py
- Data pre-processing notebook ("data_pre_precossing.ipynb")
- TabNet, XGBoost and FC neural network results notebook ("models.ipynb")
- RTDL results notebook ("rtdl.ipynb")
- XGBoost optimization notebook ("xgboost_optimizing.ipynb")
---

## How to Use
- First install TabNet to your work environment - "pip install pytorch-tabnet" 
- Create a folder for all the code files, inside it create another folder named 'data'
- Download dataset from link bellow 
- In data pre-processing notebook, delete '#' from last row in last block and run the code
- Run models.ipynb, upload 'preprocessed_data.csv'. Choose data type [original, upsampled, downsampled] and run to get results.
![image](https://user-images.githubusercontent.com/76391110/215355078-2bfda5ec-ed86-46c3-920f-0b451f909c4c.png)

note:
XGBoost optimization notebook was used to optimize XGBoost parameters (for comparison with the DL models) and does not need to be run again.

---
## Sources

### TabNet article:
https://arxiv.org/abs/1908.07442 

### Dataset from Kaggle:
https://www.kaggle.com/datasets/meirnizri/covid19-dataset

### TabNet pyTorch documentation:
https://dreamquark-ai.github.io/tabnet/

### rtdl article: 
https://arxiv.org/abs/2106.11959
