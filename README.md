# Optimizing an ML Pipeline in Azure

## Table of contents
   * [Overview](#Overview)
   * [Summary](#Summary)

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, I had the opportunity to build and optimize an Azure Machine Learning Pipeline using the Python SDK and a provided Scikit-learn logistic regression model which was customized for the project. Finally, this model is then compared to an Azure AutoML run to identify the model with the best accuracy.

The project followed the steps indicated in this image.
![Diagram](images/creating-and-optimizing-an-ml-pipeline.png)

_Step 1_: Set up the [train script](train.py), create a Tabular Dataset from the [Bank marketing dataset](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) and finally evaluate it with a Scikit-learn logistic regression model.

_Step 2_:  Create a [Jupyter Notebook](udacity-project.ipynb) and use HyperDrive to find the best hyperparameters for the logistic regression model.

_Step 3_:  Load the same [dataset](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) in the Notebook with TabularDatasetFactory and use AutoML to let Azure evaluate the model with the best fit from a selection of different models.

_Step 4_:  Finally, compare the results of the two models and write a research report.

## Summary
This project uses the [Bank marketing dataset](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing). It belongs to a portuguese banking institution and it is related to direct marketing campaigns.
The dataset has the following attribute information:
######Input variables:
**bank client data**
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
**related with the last contact of the current campaign**
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
**other attributes**
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
**social and economic context attributes**
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

######Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

**What are the benefits of the parameter sampler you chose?**

**What are the benefits of the early stopping policy you chose?**

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
