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
Based on the heatmap above, the features that are **related** to the target variable (Clicked on Ad) and will be used for modeling are **Age**, **Area Income**, **Daily Internet Usage**, and **Daily Time Spent on Site** because they have **correlation >= 0.05** with the target variable.

## 4. Data Preprocessing
- Impute null values in the **Area Income** column with **median** because it has a **skewed** distribution and **Daily Internet Usage** and **Daily Time Spent on Site** columns with **mean** because they have almost **symmetric** distributions.
- Dataset **does not have** duplicated data.
- Encode the target variable (Clicked on Ad) to numerical data, 'No': 0 and 'Yes': 1.
- Split the data into 70:30 proportions, 70% for training and 30% for testing.
- Handle outliers in the training data.





