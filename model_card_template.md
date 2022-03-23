# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Developer: Khang Nguyen \
Date: March 16th 2022  \
Version: 1.0.0 \
Type: Logistic Regression Classifier Model \
The algorithm's parameters are default based on Scikit-learn library.\
Email: khang.nguyen6461@gmail.com

## Intended Use
This model is used to classify a person make more or less than 50K per year. \
The users of this model can be anyone who want to know Income of their customers, such as: bank, real-estate salers,... \
Out-of-scope use cases: This model can not use to classify the buying power of customers; customers can pay more or less their income.

## Training Data
<p>
Name of dataset: Census Income (https://archive.ics.uci.edu/ml/datasets/census+income) <br>
We use this dataset due to these reasons: <br>
 - It is public <br>
 - It has no missing values <br>
All cateogrical features will be encoded by One-hot Encoder. <br>
There are 14 features using to train model and the size is 80% of whole dataset<br>

</p>


## Evaluation Data
<p>
Name of dataset: Census Income (https://archive.ics.uci.edu/ml/datasets/census+income) <br>
We use this dataset due to these reasons: <br>
 - It is public <br>
 - It has no missing values <br>
All cateogrical features will be encoded by One-hot Encoder. <br>
There are 14 features using to train model and the size is 20% of whole dataset<br>
</p>


## Metrics
<p>
Precision means the percentage of your results which are relevant. <br>
Precision formula is TP/(TP+FP). Precision value of model is 0.71<br>
Recall refers to the percentage of total relevant results correctly classified by your algorithm. <br>
Recall formula is TP/(TP+FN). Recall value of model is 0.25<br>
The F-beta score is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its worst value at 0. In this model, the impact of precision and recall will be the same. F-beta value is 0.37 <br>
Based on recall score, model is biased toward positive label -> the dataset is imbalanced.
</p>

## Ethical Considerations
<p> 
The model does not use any sensitive data. The model intended to inform we earn more or less 50K a year.
</p>


## Caveats and Recommendations
<p>
Given gender classes are binary (male/not male), which we include as male/female. Further work needed to evaluate across a spectrum of genders.<br>
Given race classes are binary (black/white), which we include as black/white. Further work needed to evaluate across a spectrum of skin color.
</p>
