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
import streamlit as st

"""## 1.0 Exploratory Data Analysis"""
df = pd.read_csv("Telco-Customer-Churn.csv")
"""#### 1.1 df.shape"""
st.write('Row =', df.shape[0])
st.write('Column =', df.shape[1])
"""The Telco Customer Churn dataset has 7043 rows and 21 columns."""
"""#### 1.2 df.types"""
st.write(df.dtypes)
"""This method returns a series with the data type of each column. SeniorCitizen, tenure,
MonthlyCharges are numerical data, and the rest of the attributes are categorical data."""
"""#### 1.3 df.isnull().sum()"""
st.write(df.isnull().sum())
"""This method gives the total count of null for each column (attribute)."""
"""#### 1.4 df"""
st.write(df)
"""#### 1.5 df.sample(n = 10)"""
st.write(df.sample(n=10))
"""This method returns a random sample of 10 rows (n = 10) from the dataset."""
"""#### 1.6 df.describe()"""
st.write(df.describe())
"""For numerical data, the result’s index will include count, mean, std, min, max as well as lower,
50 and upper percentiles. By default, the lower percentile is 25 and the upper percentile is 75.
The 50 percentile is the same as the median."""
 
st.markdown("<hr style='border: 10px solid #ddd;'>", unsafe_allow_html=True)

"""
## 2.0 What determines the reasons for customers to give up the services

#### The Question

1. Descriptive: A descriptive question is one that seeks to summarize a characteristic of a set of data.

2. Exploratory: An exploratory question is one in which you analyze the data to see if there are patterns, trends, or relationships between variables.

3. Inferential: An inferential question would be a restatement of the proposed hypothesis as a question and would be answered by analyzing a different set of data. The proposed hypothesis is usually derived from an exploratory question.

4. Predictive: A predictive question would be one where you ask what are the set of predictors / factors for a particular behaviour.

5. Causal: A causal question asks about whether changing one factor will change another factor, on average, in a population.

6. Mechanistic: A mechanistic question points to how a factor affects the outcome.
"""
st.markdown("---")
"""
#### 2.1 Descriptive Question
What is the churn rate derived from the dataset?
"""

# Plot the distribution of churn with customer counts
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x='Churn', data=df, palette='viridis', ax=ax)

for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, -30), textcoords='offset points', color='white', fontweight='bold')

plt.title('Churn Distribution with Customer Counts')
plt.xlabel('Churn')
plt.ylabel('Number of Customers')

# Display the plot in Streamlit
st.pyplot(fig)

"""
###### Finding 1

The churn rate is approximately 26.54%. This indicates the percentage of customers who have churned out of the total customer population.
"""
st.markdown("---")
"""
#### 2.2 Exploratory Question
How do the churn correlate with the adoption of additional services like Online Backup, Online Security, 
Streaming TV, Streaming Movies, Device Protection, and Tech Support?
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

# Display the correlation matrix plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(service_df.corr(), annot=True, cmap='rocket_r', fmt=".2f", linewidths=.5, ax=ax)
plt.title('Correlation Matrix: Churn and Additional Services')
st.pyplot(fig)

"""
###### Finding 2

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
st.markdown("---")
"""
#### 2.3 Inferential Question
Based on the observed higher churn rate for customers with a partner in the dataset, can we infer that this difference is consistent for customers with dependents?
"""

# Convert churn column to numerical values
df['churn_numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Melt the DataFrame for easier plotting
melted_df = pd.melt(df, id_vars=['Churn', 'churn_numeric'], value_vars=['Partner', 'Dependents'],
                    var_name='Attribute', value_name='Category')


# Plot churn rate by Partner and Dependents
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x='Attribute', y='churn_numeric', hue='Category', data=melted_df, palette='YlOrBr', ax=ax)
plt.title('Churn Rate by Partner and Dependents')
plt.xlabel('Attribute')
plt.ylabel('Churn Rate')
plt.legend(title='Category')
st.pyplot(fig)

# Calculate and display churn rates
partner_churn_rate = df[df['Partner'] == 'Yes']['churn_numeric'].mean()
dependents_churn_rate = df[df['Dependents'] == 'Yes']['churn_numeric'].mean()

st.write(f"Churn rate for customers with a partner: {partner_churn_rate:.2%}")
st.write(f"Churn rate for customers with dependents: {dependents_churn_rate:.2%}")

# Calculate and display churn rates for different combinations
st.write("###### Churn Rates for Different Combinations")

# Define partner and dependents combinations
partner_dependents_combinations = [
    ('Yes', 'Yes'),
    ('Yes', 'No'),
    ('No', 'Yes'),
    ('No', 'No')
]

# Create a list to store the results
churn_rates_results = []

# Calculate churn rates for different combinations
for partner_status, dependents_status in partner_dependents_combinations:
    subset_df = df[(df['Partner'] == partner_status) & (df['Dependents'] == dependents_status)]
    churn_rate = subset_df['churn_numeric'].mean() * 100  # Multiply by 100
    churn_rate_rounded = round(churn_rate, 2)  # Round to 2 decimal places
    
    # Append the results to the list
    churn_rates_results.append({
        'Partner': partner_status,
        'Dependents': dependents_status,
        'Churn Rate (%)': churn_rate_rounded
    })

# Convert the list to a DataFrame
churn_rates_df = pd.DataFrame(churn_rates_results)

# Display the churn rates as a table using Streamlit
st.table(churn_rates_df.style.format({'Churn Rate (%)': '{:.2f}%'}))

"""
###### Finding 3
- The difference in churn rates for customers with a partner appears to be influenced by the presence or absence of dependents.
- Customers with both a partner and dependents have the lowest churn rate, suggesting that having both a partner and dependents may contribute to higher customer retention.
- However, the churn rate is higher for customers with a partner but no dependents, indicating that the relationship between having a partner and churn is influenced by other factors, such as the presence of dependents.
"""
st.markdown("---")
"""
#### 2.4 Predictive Question
Can we predict the likelihood of churn for a customer based on their contract type with the company?
"""

# Count the occurrences of churn for each contract type
churn_counts = df.groupby(['Contract', 'Churn']).size().reset_index(name='Counts')

# Pivot the DataFrame for easy plotting
pivot_df = churn_counts.pivot(index='Contract', columns='Churn', values='Counts')

# Plot the churn distribution by contract type
fig, ax = plt.subplots(figsize=(8, 5))
pivot_df.plot(kind='bar', color=['lightcoral', 'skyblue'], ax=ax)
plt.title('Churn Distribution by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.legend(title='Churn', labels=['Churn', 'No Churn'], loc='upper right')
plt.xticks(rotation=0)

# Annotate the bars with counts
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', color='black', fontweight='bold')

# Display the plot in Streamlit
st.pyplot(fig)

# Count the occurrences of churn for each contract type
contract_churn_counts = df.groupby(['Contract', 'Churn']).size().reset_index(name='Counts')
pivot_df = contract_churn_counts.pivot(index='Contract', columns='Churn', values='Counts').fillna(0).astype(int)
pivot_df['Total'] = pivot_df['No'] + pivot_df['Yes']
pivot_df['Churn Rate (%)'] = (pivot_df['Yes'] / pivot_df['Total']) * 100

# Display the churn distribution by contract type as a table
st.write("###### Churn Distribution by Contract Type:")
st.write(pivot_df[['No', 'Yes', 'Total', 'Churn Rate (%)']].round(2))

"""
###### Finding 4
Customers with longer contract durations (one year and two years) tend to have lower churn rates 
compared to those with month-to-month contracts. This suggests that longer-term contracts are associated with higher customer retention.
"""
st.markdown("---")
"""
#### 2.5 Causal Question
Does the introduction of a more customer-friendly payment method, such as providing incentives for customers to switch to automatic bank transfers or credit card payments, lead to a reduction in customer churn rates?
"""

# Select relevant columns
payment_churn_df = df[['PaymentMethod', 'Churn']]

# Create a bar chart to visualize the relationship between payment method and churn
fig, ax = plt.subplots(figsize=(12, 5))
sns.countplot(x='PaymentMethod', hue='Churn', data=payment_churn_df, palette='magma', ax=ax)

for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, -25), textcoords='offset points', color='white', fontweight='bold')

plt.title('Churn Distribution by Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Number of Customers')
plt.xticks(rotation=-15, ha='left')
plt.legend(title='Churn', labels=['No Churn', 'Churn'])

# Display the plot in Streamlit
st.pyplot(fig)

# Count the occurrences of churn for each payment method
payment_churn_counts = df.groupby(['PaymentMethod', 'Churn']).size().reset_index(name='Counts')
pivot_df = payment_churn_counts.pivot(index='PaymentMethod', columns='Churn', values='Counts').fillna(0).astype(int)
pivot_df['Total'] = pivot_df['No'] + pivot_df['Yes']
pivot_df['Churn Rate (%)'] = (pivot_df['Yes'] / pivot_df['Total'])*100

# Display the churn distribution by payment method as a table
st.write("###### Churn Distribution by Payment Method:")
st.write(pivot_df[['No', 'Yes', 'Total', 'Churn Rate (%)']].round(2))


"""
###### Finding 5
- It is observed that the churn rate is significantly higher for customers using electronic check as the payment method compared to other payment methods. 
- Therefore, the results suggest that customers using electronic check tend to have a higher likelihood of churning. 
- This supports the idea that introducing more customer-friendly payment methods (e.g., bank transfer or credit card) could potentially lead to a reduction in customer churn rates.
"""
st.markdown("---")
"""
#### 2.6 Mechanistic Question
How does the length of time a customer stays with the company (tenure) impact their likelihood of churning, and can we identify specific patterns or trends in tenure that contribute to customer retention or attrition?
"""

# Create a boxplot for the distribution of tenure by Churn
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x='Churn', y='tenure', data=df, palette='Blues', ax=ax)
plt.title('Distribution of Tenure by Churn')
plt.xlabel('Churn')
plt.ylabel('Tenure (months)')

# Display the plot in Streamlit
st.pyplot(fig)

# Calculate summary statistics for tenure
summary_stats = df.groupby('Churn')['tenure'].describe().round(2)

# Display the summary statistics as a table
st.write("###### Summary Statistics for Tenure:")
st.write(summary_stats)

"""
###### Finding 6

Customers who have been with the company for a longer period (higher tenure) have a lower likelihood of churning.
Customers with a lower tenure, on average, are more likely to churn.
"""
st.markdown("---")
"""
#### Overall Conclusion
- Churn appears to be influenced by a combination of service-related factors, relationship
dynamics, contract durations, payment methods, and customer tenure.
- Offering reliable and value-added services, understanding the nuances of customer
relationships, promoting longer contract durations, and improving payment options could
collectively contribute to reducing churn.
- Regular monitoring and adaptation of strategies based on customer behavior patterns are
crucial for maintaining customer satisfaction and loyalty.
"""
