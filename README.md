# H1B-Visa-Petitions
Predicting whether an individual will be able to obtain an h1B Visa 

## Overview
H-1B visas allow U.S employers to employ foreign workers in highly technical occupations. However, not all applicants receive the immigration status and many are rejected during the certification process. 

Can we use machine learning and data analytics to predict who will be successfully granted an H-1B Visa? 

I will use both **Logistic Regression** and a **Deep Neural Network** developed in TensorFlow and compare the two methods and their respective accuracy.

The dataset can be found on [Kaggle](https://www.kaggle.com/nsharan/h-1b-visa)

## Summary Statistics 
Here are some basic statistics about the wages of the applicants

|Statistic| Prevailing Wage |
|:-------:|:---------:|
|count    |3,002,373|  
|mean     |146,998|    
|std      |5,287,609|   
|min      |0|         
|25%      |54,371|      
|50%      |65,021|    
|75%      |81,432|     
|max      |6.997607e+09|    

Here are some statistics on the qualitative variables

|Statistic| Employers | Job Title | Full Time Position | Worksite |
|:-------:|:---------:|:---------:|:------------------:|:--------:|
| Number  | 263014    |  287550   | 3                  |    18622 |
|Most Popular| INFOSYS LIMITED | PROGRAMMER ANALYST | FULL TIME | NEW YORK, NEW YORK|


## H-1B Destination Visuallization
This map shows where in the US are the applicants trying to work

![h1b map](https://github.com/nrao57/H1B-Visa-Petitions-/blob/master/h1b_map.png)

## Neural Network Training
Architecture

![nn](https://github.com/nrao57/H1B-Visa-Petitions-/blob/master/NeuralNetworkArchitecture.png)

Training

![nn](https://github.com/nrao57/H1B-Visa-Petitions-/blob/master/NeuralNetworkTraining.png)

## Conclusions 
The accuracy of the Logistic Regression Model is 0.954635108481 or **95%** 

The accuracy of a Neural Network with 2 hidden layers with 10 neurons each (a very small network) is 0.955852508545 or **96%**
