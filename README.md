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
![Univariate Analysis](image/Univariate%20Analysis%20Numerical.png)
#### Observation:
- The **more time spent** on the site or the internet, the **less likely** a customer will click on an ad.
- The **older** the customer, the **more likely** a customer will click on an ad.
- The **higher** area income of customer, the **less likely** a customer will click on an ad.

### 3.2. Bivariate Analysis
