# Optimizing an ML Pipeline in Azure

## Table of contents
   * [Overview](#Overview)
   * [Summary](#Summary)
   * [Scikit-learn Pipeline](#Scikit-learn-Pipeline)
   * [AutoML](#AutoML)
   * [Pipeline Comparison](#Pipeline-comparison)
   * [Future Work](#Future-work)
   * [References](#References)
## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure Machine Learning Pipeline using the Python SDK and a provided Scikit-learn logistic regression model. This model is then compared to an Azure AutoML run and compared to the previous model to identify the most accurate model.

The project followed the steps indicated in this architecture.
![Diagram](images/creating-and-optimizing-an-ml-pipeline.png)

## Summary
This project uses a Bank marketing dataset. It belongs to a portuguese banking institution and it is related to marketing campaigns.
this dataset has the following fields:

Input variables:

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

Output variable (desired target):

21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

**Results:**
In this case, the best performing model was found using AutoML, it used Voting Ensemble and obtained 91.6% accuracy.

The accuracy of the Scikit-learn based logistic regression using hyperdrive was 90.7%

## Scikit-learn Pipeline
**Pipeline architecture**
- The data is formatted with a Tabular Dataset using the TabularDatasetFactory. 
- The data is then cleaned
- The data is then preprocessed with the clean_data function.
- The data is then split into a 80:20 ratio for training and testing.
- Create a Scikit-learn based Logistic Regression model with the dataset.
- The hyperparameters are optimized using HyperDrive.
- The accuracy is calculated on the test set for each run 
- Finally, the best model is saved.

**Parameter sampler**

The parameter sampler was specified as such:

```Python
ps = RandomParameterSampling({
    "--C" : choice(0.01, 0.1, 1),
    "--max_iter" : choice(20, 40, 60, 80, 100, 120, 140, 160, 180, 200)
})
```
- "--C" is the Regularization
- "--max_iter" is the maximum number of iterations

RandomParameterSampling is one of the choices available for the sampler, it was chosen because it supports early termination of low-performance runs, with this sampler we are still able to find reasonably good models when compared to other sampler policies such as GridParameterSampling or BayesianParameterSampling that exhaustively searches over all the hyperparameter space.

**Early stopping policy**

The early stopping policy is used to stop poorly performing runs. Specifically, the BanditPolicy cuts more runs than other early stopping policies, that's why it was chosen.

It was run with the following configuration parameters:

```Python
policy = BanditPolicy(slack_factor = 0.1, evaluation_interval=1, delay_evaluation=5)
```

- slack_factor: The ammount specifies the allowable slack as a ratio, in the run with the highest accuracy.

- evaluation_interval: The frequency for applying the policy. It counts as one interval for each log of the primary metric by the script.

- delay_evaluation: For the a specified number of intervals delays the first policy evaluation.

## AutoML

AutoML provides the ability to automatically run multiple experiments, compare and finally choose the best performing clasification model.

**AutoML Pipeline**

- The dataset is converted with TabularDataset.
- The data is cleaned and divided into a 80:20 ratio train and test sets
- Configuration for AutoML is set. The following configuration was given:
```Python
automl_config = AutoMLConfig(
   experiment_timeout_minutes=30, #The limit of how long the experiment should run
   task="classification", #The experiment type
   primary_metric="accuracy", #The primary metric
   training_data=train_data, #The dataset to train the data
   label_column_name='y', #The objective variable
   n_cross_validations=5, #How many cross validations to perform, based on the same number of subsets.
   compute_target=compute_target #Where the processing will take place
)
```
- AutoML runs in various models
- The accuracy is calculated on the test set for each run 
- Finally, the best model is saved.

## Pipeline comparison

**Scikit-learn Pipeline**

The Scikit-learn Pipeline created one HyperDrive model with an accuracy of 90.7%

**AutoML Pipeline**

AutoML ran 18 models in total:
| ITERATION | PIPELINE                                | DURATION     | METRIC       | BEST       |
|-----------|----------                               |----------    |--------      |------      |
|0          |MaxAbsScaler LightGBM                    |0:00:57       |0.9142        |0.9142      |
|1          |MaxAbsScaler XGBoostClassifier           |0:01:06       |0.9137        |0.9142      |
|2          |MaxAbsScaler RandomForest                |0:01:10       |0.8908        |0.9142      |
|3          |MaxAbsScaler RandomForest                |0:01:04       |0.8883        |0.9142      |
|4          |MaxAbsScaler RandomForest                |0:00:58       |0.8075        |0.9142      |
|5          |MaxAbsScaler RandomForest                |0:01:07       |0.7842        |0.9142      |
|6          |SparseNormalizer XGBoostClassifier       |0:01:23       |0.9106        |0.9142      |
|7          |MaxAbsScaler GradientBoosting            |0:01:13       |0.9022        |0.9142      |
|8          |StandardScalerWrapper RandomForest       |0:01:05       |0.9010        |0.9142      |
|9          |MaxAbsScaler LogisticRegression          |0:01:08       |0.9083        |0.9142      |
|10         |MaxAbsScaler LightGBM                    |0:01:15       |0.8915        |0.9142      |
|11         |SparseNormalizer XGBoostClassifier       |0:01:15       |0.9111        |0.9142      |
|12         |MaxAbsScaler ExtremeRandomTrees          |0:04:27       |0.8883        |0.9142      |
|13         |StandardScalerWrapper LightGBM           |0:00:52       |0.8883        |0.9142      |
|14         |SparseNormalizer XGBoostClassifier       |0:01:45       |0.9117        |0.9142      |
|15         |StandardScalerWrapper ExtremeRandomTrees |0:01:05       |0.8883        |0.9142      |
|16         |StandardScalerWrapper LightGBM           |0:05:00       |0.8883        |0.9142      |
|17         |VotingEnsemble                           |0:03:39       |0.9160        |0.9160      |
|18         |StackEnsemble                            |0:01:28       |0.9152        |0.9160      |

For the AutoML Pipeline, the best model was VotingEnsemble with an accuracy of 91.6%

The difference between each model is not too big, but the AutoML model has a better accuracy, probably because AutoML makes the necessary calculations, trainings and validations automatically, and in the case of the logistic regression one has to come to a final result by trial and error, and it could be that we did not select the best configuration for our model, hence why the AutoML model had better accuracy

## Future work

Although the model had good results, there are some areas of improvement to take into consideration for the next iteration:
- In the case of the scikit-learn based model, a different parameter sampler could be used to not stop poor performance runs, this may bring marginal increases in accuracy, but will make the pipeline take more time to finish. Also, other parameters in the HyperDrive configuration can be tweaked around to optimize the pipeline.

- In case of AutoML based model, Azure gave an Imbalanced Data warning, which should be addressed for future runs to elimiate de bias the data has towards one class. Also the 30 minutes time limit could be removed to allow more time to process a certain model.

## References

- [Bank marketing dataset](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- [Random Parameter Sampling Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling?view=azure-ml-py)
- [Hyperparameter tuning a model with Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)
- [Exam DP-100 Topic 4 Question 36 Discussion](https://www.examtopics.com/discussions/microsoft/view/36687-exam-dp-100-topic-4-question-36-discussion/)
- [Azure bandit_policy documentation](https://www.rdocumentation.org/packages/azuremlsdk/versions/1.10.0/topics/bandit_policy)
