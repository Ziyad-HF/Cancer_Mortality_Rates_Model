# Cancer Mortality Rates Model

![Project Logo or Banner (optional)]

## Overview

The "Cancer Mortality Rates Model" project is a comprehensive data science and machine learning endeavor aimed at understanding the relationship between socioeconomic status and cancer mortality rates in the United States from 2010 to 2016. This project combines data analysis, preprocessing, modeling, and evaluation to derive valuable insights and build predictive models for assessing the impact of various attributes on cancer mortality rates.

## Dataset

The project utilizes a comprehensive dataset aggregated from [data.world](https://data.world/nrippner/ols-regression-challenge) website, including the American Community Survey. This dataset contains a wealth of information about US counties, including socioeconomic indicators and cancer mortality rates. For detailed information on the dataset, data sources, and preprocessing, please refer to the [Dataset](/cancer_reg.csv).

## Usage

To use the project, follow the Jupyter notebooks in the [Model](/model.ipynb) directory or Python script at [Model Script](/model_script.py). These notebooks guide you through the entire data analysis and modeling process. You can use [Google colab notebook](https://colab.research.google.com/drive/12DzknM_ri3z8X_oYgOiAe4W_QZYkgPV-?usp=sharing) which is ready with results.

## Used Libraries
We used several python libraries like pandas, matplotlib, seaborn, numpy, scipy, and statsmodels.

## Data Exploration and Analysis

This exploration helped us identify the nature of each attribute and determine what quantitative and categorical variables are. We found out that there are only two categorical variables, which are the city and district from which the data was collected.Through visualization, we also noticed the discrepancy between each county's real and expected death rates.

## Data Cleaning

The attributes that have null values and how to solve this problem. So we test 3 methnologies to test which is the best for using on our data:
1. Filling all null values with zero
2. Removing the rows that have null values
3. Filling null values with the mean of the values that are present in their column
4. Applying the forward fill method

## Handling the outliers
Visualizing the data helped us see that the data had a lot of outliers that needto be handled

## Feature Engineering

Feature engineering was a crucial step in selecting and engineering attributes for our machine learning models.After we had successfully removed the outliers, we went on to choose the attributes that were believed to strongly affect the data. At first, we calculated the Pearson correlation coefficient between each attribute of the numeric attributes and the other attributes using Pandas. Then, we made a heatmap for these calculations using the seaborn python package.

## Model Building

We experimented with various machine learning algorithms and techniques to build predictive models for cancer mortality rates. So we used the Scikit-learn package to make a multivariable regression model between the target death rate and all other features.

## Evaluation

We evaluated model performance using appropriate metrics and conducted hypothesis testing to assess the significance of our findings.

## Contributors

- [Ziyad ElFayoumy](https://github.com/Zoz-HF)
- [Mohamed Ibrahim](https://github.com/Medo072)
- [Kareem Salah](https://github.com/cln-Kafka)
- [AbdElRahman Shawky](https://github.com/AbdulrahmanGhitani)
- [Ahmed Kamal](https://github.com/AhmedKamalMohammedElSayed)

## Acknowledgments

We extend our gratitude to our supervisors, Dr. Ibrahim and Eng. Merna, for their invaluable guidance and support throughout this project.