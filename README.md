**a. Problem statement**

The goal of this project is to predict the purchasing intent of online shoppers based on their session behavior. By analyzing features such as page values, bounce rates, and visitor types, we aim to classify whether a session will result in "Revenue" (a purchase) or not. This is a binary classification problem that helps e-commerce businesses optimize their marketing strategies.



**b. Dataset description**
Source:  Kaggle

Instances: 12,330.

Features: 18 total (10 numerical, 8 categorical).

Target Variable: Revenue (True/False)

Minimum Requirements Met: The dataset met the required minimum of 12 features and 500 instances.



**c. Models used:**



|**ML Model Name**|**Accuracy**|**AUC**|**Precision**|**Recall**|**F1**|**MCC**|
|-|-|-|-|-|-|-|
|**Logistic Regression**|0.8805|0.8794|0.7472|0.3496|0.4763|0.4574|
|**Decision Tree**|0.8921|0.9207|0.7047|0.5270|0.6030|0.5497|
|**kNN**|0.8754|0.7917|0.6851|0.3670|0.4779|0.4405|
|**Naive Bayes**|0.8494|0.8364|0.5151|0.5339|0.5243|0.4350|
|**Random Forest <br />(Ensemble)**|0.898|0.9251|0.7223|0.5565|0.6287|0.5771|
|**XGBoost <br />(Ensemble)**|0.9002|0.9262|0.7173|0.5913|0.6482|0.5945|



|**ML Model Name** |**Observation about model performance**|
|-|-|
|**Logistic Regression**|This model serves as a strong baseline for accuracy (0.8805), but it shows significant difficulty in identifying actual buyers, as seen by the low Recall (0.3496). It is highly biased toward the majority class (non-purchasers).|
|**Decision Tree**|There is a notable jump in AUC (0.9207) and MCC (0.5497) compared to Logistic Regression. It handles non-linear relationships much better, nearly doubling the Recall, which is crucial for identifying potential revenue.|
|**kNN**|This performed as the weakest model in terms of AUC (0.7917) and MCC (0.4405). Since kNN relies on distance metrics, it likely struggled with the high dimensionality of the 18 features in this dataset.|
|**Naive Bayes**|While it has the lowest Accuracy (0.8494), it surprisingly maintains a better Recall (0.5339) than Logistic Regression. This suggests it is more willing to predict a "Purchase," though at the cost of many False Positives.|
|**Random Forest<br />(Ensemble)**|As an ensemble method, it provides a very stable performance with a high AUC (0.9251). It effectively reduces the variance seen in the single Decision Tree, leading to a more reliable F1-Score (0.6287).|
|**XGBoost<br />(Ensemble)**|Best Overall Performer. It achieved the highest Accuracy (0.9002), Recall (0.5913), and MCC (0.5945). Its gradient boosting mechanism successfully optimized the prediction of the minority class better than any other model.|



