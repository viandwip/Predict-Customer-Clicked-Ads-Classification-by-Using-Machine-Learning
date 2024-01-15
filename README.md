# Predict Customer Clicked Ads Classification by Using Machine Learning

## 1. Goals & Objectives
**Goals:** 
1. Increase advertising effectiveness to above 90%.<br>
2. Find out the factors that influence customers to click on ads.

**Objectives:**
1. Analyze historical advertisement data to find insights and patterns that occur.
2. Create a machine learning classification model to determine the right target customers.

## 2. Data Description
| Feature | Description | Type | 
| :- | :- | :- |
| Unnamed: 0 | ID Customers| Numeric | 
| Daily Time Spent on a Site | Time spent by the customers on a site in minutes. | Numeric | 
| Age  | Customer's age in years. | Numeric | 
| Area Income  | Average income of geographical area of costumers. | Numeric | 
| Daily Internet Usage | Time spent by customers on the internet in one day in minutes. | Numeric | 
| Male | Whether or not a constumer was male. | Categorical | 
| Timestamp | What time customers clicked on an Ad or the closed window. | Categorical | 
| Clicked on Ad  | 'No' or 'Yes' is indicated clicking on an Ad. | Categorical | 
| city | City of the costumers. | Categorical | 
| province | Province of the costumers. | Categorical | 
| category | Category of the advertisement. | Categorical |

## 3. Exploratory Data Analysis
### 3.1. Univariate Analysis
#### 3.1.1. Numerical Features 
![Univariate Analysis Numerical Features](image/Univariate%20Analysis%20Numerical.png)
#### Observation:
- The **more time spent** on the site or the internet, the **less likely** a customer will click on an ad.
- The **older** the customer, the **more likely** a customer will click on an ad.
- The **higher** area income of customer, the **less likely** a customer will click on an ad.

#### 3.1.2. Categorical Features 
![Univariate Analysis Categorical Features](image/Univariate%20Analysis%20Categorical.png)
#### Observation:
- **Perempuan** (female) has a slightly higher probability of **clicking** on Ad than Laki-laki (male).
- Each ad category has a fairly similar click ratio with the **highest** ad category clicked being **Finance** and the **lowest** being **Bank**.
- The city with the **highest** click ratio is **Serang** and the **lowest** is **Jakarta Pusat**.
- The top 3 provinces with the **highest** click ratio are **Kalimantan Selatan**, **Banten**, **Sumatra Barat**.

### 3.2. Bivariate Analysis
![Bivariate Analysis](image/Bivariate%20Analysis.png)
#### Observation:
- Age with Daily Time Spent on Site or Daily Internet Usage have a **negative correlation**. This means that the **older** the customer, the **less time** they spend on the site or the internet.
- Meanwhile, Daily Time Spent on Site and Daily Internet Usage have a **positive correlation**. This means that the **more time** spent on the internet, the **more time** will be spent on the site too.

### 3.3. Multivariate Analysis
#### 3.3.1. Pearson Correlation
![Multivariate Analysis Pearson Correlation](image/Multivariate%20Analysis%20Pearson%20Correlation.png)
#### Observation:<br>
Based on the heatmap above, there are **no features** that are **redundant** or have high correlation (>= 0.7) between them. Therefore, all features can be used for modeling. However, by using Pearson correlation, we cannot determine the relationship between features and the target variable because the **target variable is categorical data**. Therefore, we will use **PPS (Predictive Power Score)** to calculate the relationship between features and the target variable.

#### 3.3.2. Predictive Power Score
![Multivariate Analysis Predictive Power Score](image/Multivariate%20Analysis%20Predictive%20Power%20Score.png)
#### Observation:<br>
Based on the heatmap above, the features that are **related** to the target variable (Clicked on Ad) and will be used for modeling are **Age**, **Area Income**, **Daily Internet Usage**, and **Daily Time Spent on Site** because they have **predictive power score >= 0.05** with the target variable.

## 4. Data Preprocessing
- **Impute** null values in the **Area Income** column with **median** because it has a **skewed** distribution and **Daily Internet Usage** and **Daily Time Spent on Site** columns with **mean** because they have almost **symmetric** distributions.
- Dataset **does not have** duplicated data.
- **Encode** the target variable (Clicked on Ad) to **numerical data**, 'No': 0 and 'Yes': 1.
- Split the data into 70:30 proportions, **70% for training** and **30% for testing**.
- Handle **outliers** for the **Area Income** column in the training data.
- Conduct **normalization** process for the features used in the training and testing data.

## 5. Modeling
The primary metrics that will be used is **accuracy**, because the dataset has **balanced number of labels**.
### 5.1. Before Normalization
| No | Model | Acc | Prec | Recall | Time Elapsed |
| :- | :- | :- | :- | :- | :- |
| 1 | Decision Tree | 0.953333 | 0.972973 | 0.935065 | 0.599740 |
| 2 | Random Forest |	0.953333 |	0.954545 |	0.954545 | 43946.556562 |
| 3 | Extra Trees | 0.953333	| 0.979452 | 0.928571	| 37221.365716 |
| 4 | AdaBoost | 0.943333 |	0.941935 | 0.948052 |	2.004584 |
| 5 | XGBoost	| 0.940000	| 0.947368	| 0.935065	| 11219.605488 |
| 6 | KNNeighbors	| 0.680000	| 0.750000	| 0.564935	|1389.589961 |
| 7 | Logistic Regression	| 0.616667	| 0.574713	| 0.974026	| 7.244295 |

#### Observation:
- **Tree-based** models have **far better performance** than **distance-based** models.
- The **best** performance models are **Decision Tree**, **Random Forest**, and **Extra Trees** with the highest accuracy.
- The **worst** performance models are **KNNeigbors** and **Logistic Regression** with the lowest accuracy.
- The **longest** time elapsed occured on **Random Forest**, **Extra Trees** and **XGBoost** models.

### 5.2. After Normalization
| No | Model | Acc (Normalized) | Prec (Normalized) | Recall (Normalized) | Time Elapsed (Normalized) |
| :- | :- | :- | :- | :- | :- |
| 1	| KNNeighbors	| 0.956667	| 0.986207	| 0.928571	| 1154.110018 |
| 2	| Decision Tree |	0.953333	| 0.972973	| 0.935065	| 0.270079 |
| 3	| Random Forest	| 0.953333	| 0.954545	| 0.954545	| 42926.467355 |
| 4	| Extra Trees	| 0.953333	| 0.979452	| 0.928571	| 39691.231115 |
| 5	| AdaBoost	| 0.943333	| 0.941935	| 0.948052	| 1.777168 |
| 6	| XGBoost	| 0.940000	| 0.947368	| 0.935065	| 10764.285743 |
| 7 |	Logistic Regression	| 0.926667	| 1.000000	| 0.857143	| 0.189288 |

#### Observation
- After the dataset was **normalized**, there was a **significant change** in the performance of the **distance-based** models.
- The **best** performance model changes to **KNNeighbors model**. 
- The **worst** performance model still **Logistic Regression** although its performance increase significantly.
- The **longest** time elapsed still occured on **Random Forest**, **Extra Trees** and **XGBoost** models.

### 5.3. Model Comparison
| No | Model |	Acc| Acc (Norm) |	Δ Acc|	Prec |	Prec (Norm) |	Δ Prec|	Recall |	Recall (Norm) |	Δ Recall |	Time Elapsed |	Time Elapsed (Norm) |	Δ Time Elapsed |
| :- | :- | :- | :- | :- | :- | :- | :- | :- | :- | :- | :- | :- | :- |
| 1 | KNNeighbors	| 0.680000	| 0.956667	| 0.276667	| 0.750000	| 0.986207	| 0.236207	| 0.564935	| 0.928571	| 0.363636	| 1389.589961	| 1154.110018	| -235.479943 |
| 2	| Decision Tree	| 0.953333	| 0.953333	| 0.000000	| 0.972973	| 0.972973	| 0.000000	| 0.935065	| 0.935065	| 0.000000	| 0.599740	| 0.270079 | -0.329662 |
| 3	| Random Forest	| 0.953333	| 0.953333	| 0.000000	| 0.954545	| 0.954545	| 0.000000	| 0.954545	| 0.954545	| 0.000000	| 43946.556562 |	42926.467355 |	-1020.089206 |
| 4	| Extra Trees	| 0.953333	| 0.953333	| 0.000000	| 0.979452	| 0.979452	| 0.000000	| 0.928571	| 0.928571	| 0.000000	| 37221.365716 | 39691.231115	| 2469.865398 |
| 5	| AdaBoost	| 0.943333	| 0.943333	| 0.000000	| 0.941935	| 0.941935	| 0.000000	| 0.948052	| 0.948052	| 0.000000	| 2.004584	| 1.777168	| -0.227415 |
| 6	| XGBoost	| 0.940000	| 0.940000	| 0.000000	| 0.947368	| 0.947368	| 0.000000	| 0.935065	| 0.935065	| 0.000000	| 11219.605488	| 10764.285743	| -455.319745 |
| 7	| Logistic Regression	| 0.616667	| 0.926667	| 0.310000	| 0.574713	| 1.000000	| 0.425287	| 0.974026	| 0.857143	| -0.116883	| 7.244295	| 0.189288	| -7.055007 |

#### Observation:
- Overall, all models **perform better** after the dataset **normalized** based on the metrics evaluation and also the time elapsed.
- The chosen model is **Decision Tree** model because it has one of the **highest accuracy** and the **fastest computation process**.

### 5.4. Confusion Matrix
<p align="center"><img src="image/Confusion Matrix.png" alt="Confusion Matrix" width = 70%></p>
  
###
By using the results of *hyperparameter tuning* for the decision tree model, we train the model again to get a **confusion matrix** as shown above, with the following results:
- **True Positive**: Predicted to click on the ad and it turned out to be correct 144 times
- **True Negative**: Predicted not to click on the ad and it turned out to be correct 142 times
- **False Positive**: Predicted to click on the ad and turned out to be wrong by 4 times
- **False Negative**: Predicted not to click on the ad and turned out to be wrong 10 times

### 5.5. Feature Importances
![Feature Importances](image/Feature%20Importances.png)

Based on the feature importances in the image above, we can see that **Daily Time Spent on Site** is the most important feature, followed by the **Daily Internet Usage** feature in second place which determine whether **users click on ads or not**.

## 6. Business Recommendation & Simulation
### 6.1. Business Recommendation
Based on the **insight from EDA** and **feature importances**, we can provide business recommendations such as:
****

- **Content Optimization**<br>
Because the higher **Daily Time Spent on Site** and **Daily Internet Usage** the less likely user will click on ads, then we need create ad contents that are **engaging** and **relevant** to the target user and ensure that the messaging and visuals of the ads **align with the interests and needs** of the user.<br>

- **Targeted Pricing Strategies**<br>
Because the **lower** Area Income the **more likely** user will click on ads, we can implement targeted pricing strategies that **align with the income levels** of the target audience. This may involve creating **special pricing tiers**, **discounts**, or **bundled offerings**. Consider developing and promoting **affordable products** or **services** for the users with low area income.<br>

- **Age-Targeted Marketing Campaigns**<br>
Because the **older** the user the **more likely** user will click on ads, then we can develop targeted marketing campaigns specifically designed to resonate with **older demographics**. We can create the messages, visuals, and offers to align with the **preferences** and **interests** of older users.

### 6.2. Business Simulation
**Assumption:**

Cost per Mille (CPM) = Rp.100,000

Revenue per Ad Clicked  = Rp.2,000
****

**Before Using Machine Learning Model:**

- **Number of Users Advertised**:<br>
User = 1,000
- **Click-Through Rate (CTR)**: <br>
500/1,000 = 0.5
- **Total Cost**: <br>
CPM = Rp.100,000
- **Total Revenue**: <br>
CTR x Number of Users Advertised x Revenue per Ad Clicked = 0.5 x 1,000 x 2,000 = Rp.1,000,000
- **Total Profit**:<br>
Total Revenue - Total Cost = **Rp.900,000**
****

**After Using Machine Learning Model:**

- **Number of Users Advertised**:<br>
User = 1,000
- **Click-Through Rate (CTR)**: <br>
Precision = 0.95
- **Total Cost**: <br>
CPM = Rp.100,000
- **Total Revenue**: <br>
CTR x Number of Users Advertised x Revenue per Ad Clicked = 0.95 x 1,000 x 2,000 = Rp.1,900,000
- **Total Profit**:<br>
Total Revenue - Total Cost = **Rp.1,800,000**
****

**Conclusion:**<br>

From the results above, it can be seen that after we used the machine learning model, the ad performance increased. **Click-Through Rate (CTR)** increased 45% from **50% to 95%** and **total profit** increased 100% from **Rp.900,000** to **Rp.1,800,000**.






