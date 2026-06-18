# %% [markdown]
# # D2C Customer Churn - Exploratory Data Analysis & Audit
# *Note: This script represents the required `eda_audit.ipynb` notebook. Run this in an interactive Jupyter environment.*

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# %% [markdown]
# ## 1. Data Loading

# %%
customers = pd.read_csv('customers.csv')
orders = pd.read_csv('orders.csv')
tickets = pd.read_csv('support_tickets.csv')
web_events = pd.read_csv('web_events_snapshot.csv')
churn = pd.read_csv('churn_labels.csv')
interventions = pd.read_csv('intervention_history.csv')

print(f"Customers loaded: {len(customers)}")
print(f"Orders loaded: {len(orders)}")

# %% [markdown]
# ## 2. Data Quality Audit & Cleaning

# %%
# A. Handle Leakage: Remove post-snapshot orders
SNAPSHOT_DATE = pd.to_datetime('2025-09-30')
orders['order_date'] = pd.to_datetime(orders['order_date'])

initial_order_count = len(orders)
orders_clean = orders[orders['order_date'] <= SNAPSHOT_DATE].copy()
print(f"Removed {initial_order_count - len(orders_clean)} post-snapshot orders to prevent leakage.")

# %%
# B. Handle Duplicates: Clean order_id suffix _DUP
dup_mask = orders_clean['order_id'].str.endswith('_DUP', na=False)
print(f"Found {dup_mask.sum()} duplicate-flagged orders.")

orders_clean['order_id'] = orders_clean['order_id'].str.replace('_DUP', '')
orders_clean = orders_clean.drop_duplicates(subset=['order_id'], keep='first')

# %%
# C. Handle Missing Values
print("Missing values in customers:")
print(customers[['loyalty_tier', 'skin_type']].isnull().sum())
customers['loyalty_tier'] = customers['loyalty_tier'].fillna('None')
customers['skin_type'] = customers['skin_type'].fillna('Unknown')

print("Missing values in order ratings:")
print(orders_clean['rating'].isnull().sum())
orders_clean['rating'] = orders_clean['rating'].fillna(orders_clean['rating'].median())

# %%
# D. Handle Outliers in Orders
plt.figure(figsize=(8, 4))
sns.boxplot(x=orders_clean['gross_amount'])
plt.title("Boxplot of Gross Amount (Showing Outliers)")
plt.show()

# Cap outliers at 99th percentile
p99 = orders_clean['gross_amount'].quantile(0.99)
orders_clean['gross_amount'] = np.where(orders_clean['gross_amount'] > p99, p99, orders_clean['gross_amount'])

# %% [markdown]
# ## 3. Merging Data for EDA

# %%
# Merge base customer data with churn labels
df = customers.merge(churn[['customer_id', 'churn_next_60d', 'split']], on='customer_id', how='left')

# Aggregate web events
df = df.merge(web_events, on='customer_id', how='left')

# Aggregate support tickets
tickets_agg = tickets.groupby('customer_id').agg(
    total_tickets=('ticket_id', 'count'),
    avg_resolution_hours=('resolution_hours', 'mean'),
    avg_sentiment=('sentiment_score', 'mean')
).reset_index()

df = df.merge(tickets_agg, on='customer_id', how='left')
df['total_tickets'] = df['total_tickets'].fillna(0)

# %% [markdown]
# ## 4. Exploratory Data Analysis & Churn Hypotheses

# %%
# Hypothesis 1: Web Engagement Risk
# Customers with fewer 30-day sessions are more likely to churn
plt.figure(figsize=(8, 5))
sns.boxplot(x='churn_next_60d', y='sessions_30d', data=df)
plt.title("Hypothesis 1: 30-Day Web Sessions vs Churn")
plt.xlabel("Churned (1=Yes, 0=No)")
plt.ylabel("Sessions in Last 30 Days")
plt.show()

# %%
# Hypothesis 2: Support Experience Risk
# Customers with poor sentiment and long resolution hours churn more.
plt.figure(figsize=(8, 5))
sns.scatterplot(x='avg_resolution_hours', y='avg_sentiment', hue='churn_next_60d', data=df, alpha=0.6)
plt.title("Hypothesis 2: Resolution Hours & Sentiment vs Churn")
plt.xlabel("Average Resolution Hours")
plt.ylabel("Average Sentiment Score")
plt.axvline(24, color='red', linestyle='--', label='24h SLA')
plt.legend()
plt.show()

# %%
# Hypothesis 3: Lack of Loyalty Tier implies higher churn
plt.figure(figsize=(8, 5))
sns.barplot(x='loyalty_tier', y='churn_next_60d', data=df, errorbar=None)
plt.title("Hypothesis 3: Churn Rate by Loyalty Tier")
plt.ylabel("Average Churn Rate")
plt.show()

# %%
# Hypothesis 4: High last_visit_days_ago strongly predicts churn
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='last_visit_days_ago', hue='churn_next_60d', multiple="stack", bins=20)
plt.title("Hypothesis 4: Days Since Last Visit Distribution by Churn")
plt.xlabel("Days Since Last Visit")
plt.show()

# %%
# Summary prints for Business Memo
print(f"Overall Churn Rate: {df['churn_next_60d'].mean():.2%}")