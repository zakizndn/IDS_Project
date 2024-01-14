# -*- coding: utf-8 -*-
"""project3.ipynb

# Project 3

Use the following Telco dataset : [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

and perform the following tasks:

(1) EDA

(2) What determines the reasons for customers to give up the services

Attribute        | Description
-----------------|-------------------------------------------
customerID       | Customer ID
gender           | Whether the customer is a male or a female
SeniorCitizen    | Whether the customer is a senior citizen or not (1, 0)
Partner          | Whether the customer has a partner or not (Yes, No)
Dependents       | Whether the customer has dependents or not (Yes, No)
tenure           | Number of months the customer has stayed with the company
PhoneService     | Whether the customer has a phone service or not (Yes, No)
MultipleLines    | Whether the customer has multiple lines or not (Yes, No, No phone service)
InternetService  | Customer’s internet service provider (DSL, Fiber optic, No)
OnlineSecurity   | Whether the customer has online security or not (Yes, No, No internet service)
OnlineBackup     | Whether the customer has online backup or not (Yes, No, No internet service)
DeviceProtection | Whether the customer has device protection or not (Yes, No, No internet service)
TechSupport      | Whether the customer has tech support or not (Yes, No, No internet service)
StreamingTV      | Whether the customer has streaming TV or not (Yes, No, No internet service)
StreamingMovies  | Whether the customer has streaming movies or not (Yes, No, No internet service)
Contract         | The contract term of the customer (Month-to-month, One year, Two year)
PaperlessBilling | Whether the customer has paperless billing or not (Yes, No)
PaymentMethod    | The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
MonthlyCharges   | The amount charged to the customer monthly
TotalCharges     | The total amount charged to the customer
Churn            | Whether the customer churned or not (Yes or No)
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from scipy import stats

"""### (1) EDA"""
df = pd.read_csv("Telco-Customer-Churn.csv")
"""##### df.shape"""
print('Row =', df.shape[0])
print('Column =', df.shape[1])
"""##### df.types"""
df.dtypes
"""##### df.info()"""
df.info()
"""##### df.isnull().sum()"""
df.isnull().sum()
"""##### df"""
df
"""##### df.sample(n = 10)"""
df.sample(n = 10)
"""##### df.describe()"""
df.describe()

"""The Question

1. Descriptive: A descriptive question is one that seeks to summarize a characteristic of a set of data.

2. Exploratory: An exploratory question is one in which you analyze the data to see if there are patterns, trends, or relationships between variables.

3. Inferential: An inferential question would be a restatement of the proposed hypothesis as a question and would be answered by analyzing a different set of data. The proposed hypothesis is usually derived from an exploratory question.

4. Predictive: A predictive question would be one where you ask what are the set of predictors / factors for a particular behaviour.

5. Causal: A causal question asks about whether changing one factor will change another factor, on average, in a population.

6. Mechanistic: A mechanistic question points to how a factor affects the outcome.
"""

"""
Descriptive Question
What is the churn rate derived from the dataset?
"""

# Plot the distribution of churn with customer counts
plt.figure(figsize=(5, 4))
ax = sns.countplot(x='Churn', data=df, palette='viridis')

for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, -30), textcoords='offset points', color='white', fontweight='bold')

plt.title('Churn Distribution with Customer Counts')
plt.xlabel('Churn')
plt.ylabel('Number of Customers')
plt.show()

# Count the occurrences of churn
churn_counts = df['Churn'].value_counts()

for churn, count in churn_counts.items():
    print(f'Churn: {churn}, Number of Customers: {count}')

print(f'Total Number of Customers: {df.shape[0]}')

"""
Conclusion

The churn rate is approximately
26.51%. This indicates the percentage of customers who have churned out of the total customer population.
"""

"""
Exploratory Question
How do the churn correlate with the adoption of additional services like Online Security, Streaming TV, and Device Protection?
"""

# Select relevant columns for analysis and create a copy
selected_columns = ['Churn', 'OnlineBackup', 'OnlineSecurity', 'StreamingTV',
                    'StreamingMovies', 'DeviceProtection', 'TechSupport']
service_df = df[selected_columns].copy()

# Convert categorical variables to numerical for correlation analysis using .apply
service_df.loc[:, 'Churn'] = service_df['Churn'].apply(lambda x: 0 if x == 'No' else (1 if x == 'Yes' else 0))
service_df.loc[:, 'OnlineBackup'] = service_df['OnlineBackup'].apply(lambda x: 0 if x == 'No' else (1 if x == 'Yes' else 0))
service_df.loc[:, 'OnlineSecurity'] = service_df['OnlineSecurity'].apply(lambda x: 0 if x == 'No' else (1 if x == 'Yes' else 0))
service_df.loc[:, 'StreamingTV'] = service_df['StreamingTV'].apply(lambda x: 0 if x == 'No' else (1 if x == 'Yes' else 0))
service_df.loc[:, 'StreamingMovies'] = service_df['StreamingMovies'].apply(lambda x: 0 if x == 'No' else (1 if x == 'Yes' else 0))
service_df.loc[:, 'DeviceProtection'] = service_df['DeviceProtection'].apply(lambda x: 0 if x == 'No' else (1 if x == 'Yes' else 0))
service_df.loc[:, 'TechSupport'] = service_df['TechSupport'].apply(lambda x: 0 if x == 'No' else (1 if x == 'Yes' else 0))

# Plot correlation matrix
correlation_matrix = service_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='rocket_r', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix: Churn and Additional Services')
plt.show()

"""
Conclusion

Churn and Online Backup:
- There is a weak negative correlation (-0.08) between 'Churn' and 'Online Backup'.
- This suggests that customers with online backup are slightly less likely to churn.

Churn and Online Security:
- There is a weak negative correlation (-0.17) between 'Churn' and 'Online Security'.
- Customers with online security are slightly less likely to churn.

Churn and Streaming TV:
- There is a weak positive correlation (0.06) between 'Churn' and 'Streaming TV'.
- This suggests a minor positive relationship, but the correlation is not strong.

Churn and Streaming Movies:
- There is a weak positive correlation (0.06) between 'Churn' and 'Streaming Movies'.
- Similar to Streaming TV, there is a minor positive relationship, but the correlation is weak.

Churn and Device Protection:
- There is a weak negative correlation (-0.07) between 'Churn' and 'Device Protection'.
- Customers with device protection are slightly less likely to churn.

Churn and Tech Support:
- There is a moderate negative correlation (-0.16) between 'Churn' and 'Tech Support'.
- Customers with tech support are somewhat less likely to churn.
"""

"""
Inferential Question
Based on the observed higher churn rate for customers with a partner in the dataset, 
can we infer that this difference is consistent for customers with dependents?
"""

# Convert churn column to numerical values
df['churn_numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Melt the DataFrame for easier plotting
melted_df = pd.melt(df, id_vars=['Churn', 'churn_numeric'], value_vars=['Partner', 'Dependents'],
                    var_name='Attribute', value_name='Category')

plt.figure(figsize=(8, 5))
sns.barplot(x='Attribute', y='churn_numeric', hue='Category', data=melted_df, palette='YlOrBr')
plt.title('Churn Rate by Partner and Dependents')
plt.xlabel('Attribute')
plt.ylabel('Churn Rate')
plt.legend(title='Category')

plt.show()

# Calculate churn rate for customers with a partner
partner_churn_rate = df[df['Partner'] == 'Yes']['churn_numeric'].mean()

# Calculate churn rate for customers with dependents
dependents_churn_rate = df[df['Dependents'] == 'Yes']['churn_numeric'].mean()

# Print the results
print(f"Churn rate for customers with a partner: {partner_churn_rate:.2%}")
print(f"Churn rate for customers with dependents: {dependents_churn_rate:.2%}")

# Calculate churn rates for different combinations
partner_dependents_combinations = [
    ('Yes', 'Yes'),
    ('Yes', 'No'),
    ('No', 'Yes'),
    ('No', 'No')
]

for partner_status, dependents_status in partner_dependents_combinations:
    subset_df = df[(df['Partner'] == partner_status) & (df['Dependents'] == dependents_status)]
    churn_rate = subset_df['churn_numeric'].mean()
    print(f"Churn rate for customers with Partner = {partner_status} and Dependents = {dependents_status}: {churn_rate:.2%}")

"""
Conclusion

- The difference in churn rates for customers with a partner appears to be influenced by the presence or absence of dependents.
- Customers with both a partner and dependents have the lowest churn rate, suggesting that having both a partner and dependents may contribute to higher customer retention.
- However, the churn rate is higher for customers with a partner but no dependents, indicating that the relationship between having a partner and churn is influenced by other factors, such as the presence of dependents.
"""

"""
Predictive Question
Can we predict the likelihood of churn for a customer based on their contract type with the company?
"""

# Count the occurrences of churn for each contract type
churn_counts = df.groupby(['Contract', 'Churn']).size().reset_index(name='Counts')

# Pivot the DataFrame for easy plotting
pivot_df = churn_counts.pivot(index='Contract', columns='Churn', values='Counts')

ax = pivot_df.plot(kind='bar', color=['lightcoral', 'skyblue'], figsize=(12, 5))
plt.title('Churn Distribution by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.legend(title='Churn', labels=['Churn', 'No Churn'], loc='upper right')
plt.xticks(rotation=0)

for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', color='black', fontweight='bold')

plt.show()

# Count the occurrences of churn for each contract type
churn_counts = df.groupby(['Contract', 'Churn']).size().reset_index(name='Counts')
pivot_table = churn_counts.pivot(index='Contract', columns='Churn', values='Counts').fillna(0).astype(int)
pivot_table['Total'] = pivot_table.sum(axis=1)
print(pivot_table)

"""
Conclusion

Contract Type  | Churn Rate                 
---------------|----------------------
Month-to-month | 1,655 / 3,875 = ~42.7%
One-Year       | 166 / 1,473 = ~11.3%
Two-Year       | 48 / 1,695 = ~2.8%

Customers with longer contract durations (one year and two years) tend to have lower churn rates 
compared to those with month-to-month contracts. This suggests that longer-term contracts are associated with higher customer retention.
"""

"""
Causal Question
Does the introduction of a more customer-friendly payment method, such as providing incentives for customers 
to switch to automatic bank transfers or credit card payments, lead to a reduction in customer churn rates?
"""

# Select relevant columns
payment_churn_df = df[['PaymentMethod', 'Churn']]

# Create a bar chart to visualize the relationship between payment method and churn
plt.figure(figsize=(12, 5))
ax = sns.countplot(x='PaymentMethod', hue='Churn', data=payment_churn_df, palette='magma')

for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, -25), textcoords='offset points', color='white', fontweight='bold')

plt.title('Churn Distribution by Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Number of Customers')
plt.xticks(rotation=-15, ha='left')
plt.legend(title='Churn', labels=['No Churn', 'Churn'])
plt.show()

# Count the occurrences of churn for each payment method
payment_churn_counts = df.groupby(['PaymentMethod', 'Churn']).size().reset_index(name='Counts')
pivot_df = payment_churn_counts.pivot(index='PaymentMethod', columns='Churn', values='Counts')
pivot_df['Total'] = pivot_df['No'] + pivot_df['Yes']
print("Churn Distribution by Payment Method:")
print(pivot_df[['No', 'Yes', 'Total']])

"""
Conclusion

Payment Method            | Churn rate                   
--------------------------|-------------------------
Bank transfer (automatic) | 258 / 1544 = ~16.7%
Credit card (automatic)   | 232 / 1522 = ~15.2%
Electronic check          | 1071 / 2365 = ~45.3%
Mailed check              | 308 / 1612 = ~19.1%

It is observed that the churn rate is significantly higher for customers using electronic check as the payment method compared to other payment methods. Therefore, the results suggest that customers using electronic check tend to have a higher likelihood of churning. This supports the idea that introducing more customer-friendly payment methods (e.g., bank transfer or credit card) could potentially lead to a reduction in customer churn rates.
"""

"""
Mechanistic Question
How does the length of time a customer stays with the company (tenure) impact their likelihood of churning, 
and can we identify specific patterns or trends in tenure that contribute to customer retention or attrition?
"""

# Create a boxplot for the distribution of tenure by Churn
plt.figure(figsize=(6, 5))
sns.boxplot(x='Churn', y='tenure', data=df, palette='Blues')
plt.title('Distribution of Tenure by Churn')
plt.xlabel('Churn')
plt.ylabel('Tenure (months)')
plt.show()

# Calculate summary statistics for tenure
summary_stats = df.groupby('Churn')['tenure'].describe().round(2)
print(summary_stats)

"""
For customers who did not churn (Churn = No):
- The mean tenure is approximately 37.57 months.
- This suggests that, on average, customers who did not churn have been with the company for a longer period.

For customers who churned (Churn = Yes):
- The mean tenure is approximately 17.98 months.
- This implies that, on average, customers who churned have a shorter tenure with the company.

Conclusion

Customers who have been with the company for a longer period (higher tenure) have a lower likelihood of churning.
Customers with a lower tenure, on average, are more likely to churn.
"""
