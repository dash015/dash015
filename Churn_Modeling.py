#!/usr/bin/env python
# coding: utf-8

# # <center>💻 MACHINE LEARNING 💻</center>
# # <center>📈 Bank Customer Churn Prediction: Understanding Customer Retention 📊</center>
# ## <center>Submitted by: Dasharath Gholap</center>

# ### Problem Statement
# In the competitive landscape of the banking industry, maintaining customer loyalty and reducing churn rates are crucial for sustained business success. The problem statement aims to develop predictive models that leverage customer attributes to forecast whether a bank customer is likely to stay or churn. By identifying key factors influencing customer retention, the project seeks to empower banks with actionable insights to proactively engage with at-risk customers, tailor retention strategies, and ultimately enhance customer satisfaction and loyalty.

# ### Project Overview
# 
# **This project aims to predict bank customer churn by analyzing various attributes such as demographics, account details, and transaction history. It utilizes machine learning techniques to identify customers likely to leave, enabling targeted strategies for improving retention.**
# 
# ### Detailed Methodology
# 
# 1. **Data Preprocessing**: Cleansing the dataset by removing irrelevant features, imputing missing values, and encoding categorical variables for analysis.
# 
# 2. **Exploratory Data Analysis (EDA)**: Conducting statistical analyses and visualizations to understand data distributions, identify correlations, and spot outliers affecting churn.
# 
# 3. **Feature Engineering**: Creating new features and selecting significant ones using statistical methods to enhance model accuracy.
# 
# 4. **Modeling**: Deploying various algorithms such as Logistic Regression, Random Forest, and XGBoost. Models are trained and tested to forecast customer churn.
# 
# 5. **Evaluation**: Model performance is evaluated using metrics like accuracy, precision, recall, F1-score, and ROC curves to determine the best-fit model.
# 
# 6. **Hyperparameter Tuning**: Fine-tuning model parameters through methods like GridSearchCV to maximize prediction accuracy.

# ### Libraries Overview
# 
# 1. **Pandas (pd)**: Pandas is a powerful data manipulation and analysis library in Python, offering data structures and functions for efficiently handling structured data such as tabular and time-series data.
# 
# 2. **NumPy (np)**: NumPy is a fundamental package for scientific computing in Python, providing support for multidimensional arrays and matrices, along with a wide range of mathematical functions to operate on these arrays.
# 
# 3. **Seaborn (sns)**: Seaborn is a Python data visualization library based on Matplotlib, offering a high-level interface for creating attractive and informative statistical graphics.
# 
# 4. **Matplotlib.pyplot (plt)**: Part of the Matplotlib library, Matplotlib.pyplot provides a procedural interface to the Matplotlib object-oriented plotting library, allowing users to create static, animated, and interactive visualizations in Python.
# 
# 5. **Scikit-learn (sklearn)**: Scikit-learn is a comprehensive machine learning library in Python, offering a wide range of supervised and unsupervised learning algorithms, along with tools for model evaluation, preprocessing, and feature selection.
# 
# 6. **Statsmodels.stats.outliers_influence variance_inflation_factor**: This module from the Statsmodels library provides functions for detecting multicollinearity in regression analysis, with variance inflation factor (VIF) being a key metric used to quantify multicollinearity.

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Importing Dataset
# **The dataset for exploration, modeling, and interpretability, explainability is called "Churn Modeling Dataset" to be found at the kaggle.com.**
# 
# **The dataset is loaded into a pandas DataFrame df from a CSV file named 'churn_modelling.csv'.**
# 
# **This data set contains details of a bank's customers and the target variable is a binary variable reflecting the fact whether the customer left the bank (closed his account) or he continues to be a customer.**

# In[2]:


df = pd.read_csv("D:/Projects/Churn Modelling/Churn_Modelling.csv")


# In[3]:


#displaying the contents of the dataset
df


# **The dataset for churn prediction includes a range of customer attributes such as geography, gender, age, tenure, balance, and product usage, aimed at identifying factors contributing to customer churn. With 10,000 rows and 14 columns, it provides a comprehensive view of customer interactions and outcomes, with the 'Exited' column indicating whether a customer has churned, serving as the target variable for predictive modeling.**

# ### Explanatory Data Analysis
# **Perform Explanatory Data Analysis (EDA) / indicate how features correlate among themselves, with emphasis to the target/label one**

# In[4]:


# Checking the shape of the dataset
df.shape


# In[5]:


# Drop irrelevant columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)


# **These fields are unique to each customer and do not hold predictive power or relevant information for analyzing customer behavior, trends, or attributes that influence the decision to leave the bank.**

# In[6]:


# Checking summary of the dataset
print("Data Information:")
print(df.info())


# In[7]:


# Viewing stastical summary of the dataset
print("\nSummary Statistics:")
df.describe().T


# **The dataset's statistical summary reveals key insights: Credit scores range widely with an average of 650.53, suggesting varied creditworthiness among customers. The age distribution shows customers are typically middle-aged, with an average age of 38.92. Account balances vary significantly, with some customers holding no balance and others up to 250,898.09, indicating diverse financial standings. The majority of customers have a credit card and about half are active members. Finally, a 20% churn rate highlights the importance of identifying factors contributing to customer departure.**

# In[8]:


# Identifying duplicates in the dataset
duplicate = df.duplicated() 
print(duplicate.sum()) 
df[duplicate]


# In[9]:


# Checking for missing values in the dataset
df.isnull().sum()


# In[10]:


# Checking unique values in categorical features
print("\nUnique Values in Categorical Features:")
for col in df.select_dtypes(include=['object']).columns:
    print(col, ":", df[col].unique())


# **The dataset contains two categorical features, Geography and Gender, with customers distributed across France, Spain, and Germany, and identified as either Female or Male.**

# In[11]:


# Checking for target value distribution in the dataset
print("\nTarget Distribution:")
print(df['Exited'].value_counts())


# **The target distribution shows a significant imbalance, with 7,963 customers not exiting (0) and 2,037 exiting (1), indicating that around 20% of the customers in the dataset have churned.**

# ### Data Preprocessing
# **The steps collectively form an essential part of data preprocessing, which aims to clean, transform, and prepare the data for model training and evaluation**.
# **Numeric feature scaling ensures that all variables contribute equally to the model by standardizing their scale. Categorical feature encoding transforms text into a machine-readable format, allowing models to understand and use this information effectively. Splitting the data into training and testing sets enables model training on one subset and evaluation on another to assess generalization performance. This structured approach enhances model accuracy and reliability.**

# In[12]:


# Separate features (X) and target variable (y)
X = df.drop('Exited', axis=1)
y = df['Exited']


# In[13]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


# Define numeric features and categorical features
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
categorical_features = ['Geography', 'Gender']

# Define preprocessing steps for numeric features: scaling
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# Define preprocessing steps for categorical features: one-hot encoding
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine numeric and categorical preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# ### Covariance Matrix
# 
# **The covariance matrix of the dataset is computed and printed. Covariance indicates the direction of the linear relationship between variables.**

# In[15]:


# Compute covariance matrix for numeric features
covariance_matrix = np.cov(X_train[numeric_features].T)
print('\nCovariance Matrix:')
print(covariance_matrix)


# In[16]:


# Covariance Matrix Visualization
plt.figure(figsize=(18,8))
sns.heatmap(covariance_matrix, annot=True, fmt='.2f', cmap = 'coolwarm', xticklabels=numeric_features, 
            yticklabels=numeric_features)
plt.title('Covariance Matrix')
plt.show()


# **The covariance matrix visualized through the heatmap indicates the degree to which numerical variables in the dataset change together. A high positive value, such as that seen between 'Balance' and 'EstimatedSalary' suggests a stronger positive linear relationship where they increase together. Conversely, near-zero values suggest very weak linear relationships. Covariances near zero across many variables, as seen in the matrix, suggest that features do not have strong linear dependencies, which is beneficial for certain types of predictive modeling as it reduces multicollinearity concerns.**

# ### Data Visualization

# In[17]:


# Visualize the distributions of numeric features
plt.figure(figsize=(12,8))
for i, feature in enumerate(numeric_features):
    plt.subplot(3, 3, i+1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# **The histograms of numeric features from the dataset reveal a variety of distributions:**<br>
# -CreditScore appears normally distributed, indicating diverse financial health across customers.<br>
# -Age shows a right-skewed distribution, with a younger customer base being more common.<br>
# -Tenure is fairly uniform, suggesting no specific duration dominates customer banking relationships.<br>
# -Balance has a large peak at zero, indicating a significant number of customers have no balance, with the remainder displaying a broad distribution.<br>
# -NumOfProducts shows most customers have 1 or 2 products, with few opting for 3 or 4.<br>
# -HasCrCard indicates most customers possess a credit card.<br>
# -IsActiveMember has a bimodal distribution, implying a near even split between active and inactive members.<br>
# -EstimatedSalary is uniformly distributed, showing no particular salary range dominance among customers.

# In[18]:


# Plot box plots for each numeric feature
plt.figure(figsize=(12,8))
for i, feature in enumerate(numeric_features):
    plt.subplot(3, 3, i+1)
    sns.boxplot(data=df, y=feature)
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()


# **The boxplots for each numeric feature show the distribution and presence of outliers within the dataset.**<br> - The 'CreditScore' appears quite symmetric without extreme outliers.<br> - 'Age' shows a concentration between 30s to 50s, with some outliers on the upper range indicating a few older customers. <br>- 'Balance' has a substantial number of customers with low or zero balances, and the 'NumOfProducts' has outliers at the higher end. <br>- 'EstimatedSalary' is evenly distributed across the quartiles. <br>These visualizations are key for understanding the range and distribution of each feature and for identifying any potential outliers that may affect the predictive modeling.

# In[19]:


# Explore the distribution of categorical features
plt.figure(figsize=(8,4))
for i, feature in enumerate(categorical_features):
    plt.subplot(1, 2, i+1)
    sns.countplot(data=df, x=feature)
    plt.title(f'Count of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()


# The bar charts for categorical features show the distribution among the dataset's customers. The majority are from France, followed by Spain and Germany. For gender, there is a slight imbalance, with males being more than females.

# In[20]:


# Compute and visualize the correlation matrix for numeric features
numeric_df = df.select_dtypes(include = ['int64', 'float64'])
correlation_matrix = numeric_df.corr()

plt.figure(figsize = (8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# **The heatmap shows the correlation coefficients between numeric features.**<br>- Age has a moderate positive correlation with the target variable 'Exited', suggesting older customers are more likely to churn.<br>- 'NumOfProducts' and 'IsActiveMember' have negative correlations with 'Exited', indicating customers with more products or active memberships are less likely to leave. <br>- Other variables show low correlation with the target, implying no strong linear relationships with customer churn. <br>This information can inform feature selection and the understanding of factors that may influence churn.

# In[21]:


# Plot the distribution of the target variable (Exited)
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Exited')
plt.title('Distribution of Exited')
plt.xlabel('Exited')
plt.ylabel('Count')
plt.show()


# The bar chart of the 'Exited' variable illustrates an imbalanced distribution between the two classes: the majority of customers (represented by '0') have not exited, while a smaller proportion (represented by '1') have.

# In[22]:


# Explore the relationship between the target variable and other features
plt.figure(figsize=(12,8))
for i, feature in enumerate(numeric_features):
    plt.subplot(3, 3, i+1)
    sns.violinplot(data=df, x='Exited', y=feature)
    plt.title(f'Exited vs {feature}')
plt.tight_layout()
plt.show()


# **The violin plots illustrate differences in numeric feature distributions between customers who exited and those who didn't.**<br>- The plots for 'Age' and 'Balance' indicate that customers who exited tend to be older and have higher bank balances, respectively.<br>- 'NumOfProducts' shows that customers with fewer products are more likely to churn.<br>- 'IsActiveMember' suggests that inactive members are more likely to leave the bank.<br>- The 'CreditScore', 'Tenure', 'HasCrCard', and 'EstimatedSalary' distributions appear relatively similar for both groups, indicating these features might have less impact on the likelihood of exiting.

# ### Feature Engineering
# **Variance Inflation Factor (VIF) is computed for numeric features to check for multicollinearity.**

# In[23]:


# Compute VIF for numeric features
vif_data = df[numeric_features]
vif = pd.DataFrame()
vif['Feature'] = vif_data.columns
vif['VIF'] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
print(vif)


# The Variance Inflation Factor (VIF) results help in assessing which features may introduce multicollinearity concerns that could skew the interpretation of model coefficients. While high VIFs for 'CreditScore' and 'Age' suggest possible multicollinearity, this is more relevant in regression analysis than classification.

# ### Model Training and Evaluation

# In[24]:


# Define the classifiers
classifiers = {
    'Logistic Regression' : LogisticRegression(),
    'Support Vector Machine' : SVC(probability = True),
    'SVM with Gaussian Kernel' : SVC(kernel= 'rbf', probability = True),
    'Decision Tree' : DecisionTreeClassifier(),
    'Extra Trees' : ExtraTreesClassifier(),
    'Random Forest' : RandomForestClassifier(),
    'Gradient Boosting' : GradientBoostingClassifier(),
    'LightGBM' : LGBMClassifier(),
    'K-Nearest Neighbors' : KNeighborsClassifier(),
    'Naive Bayes' : GaussianNB(),
    'XGBoost' : XGBClassifier(),
    'CatBoost' : CatBoostClassifier(),
    'AdaBoost' : AdaBoostClassifier()
}

# Train and evaluate each classifier
for name, classifier in classifiers.items():
    model = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    print(f'\n{name} (with Feature Selection):\n')
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('\nClassification Report:\n', classification_report(y_test, y_pred))
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Plot ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {name}')
    plt.legend(loc='lower right')
    plt.show()


# **When evaluating the performance of classification models, there are several key metrics to consider:**
# 
# 1.**Accuracy**: This gives you the overall correctness of the model but can be misleading if the classes are imbalanced.
# 
# 2.**Precision**: Indicates the ratio of true positives to both true and false positives. It's about how many of the positively predicted cases actually were positive.
# 
# 3.**Recall (Sensitivity)**: The ratio of true positives to true positives and false negatives. This tells you how many of the actual positives the model correctly predicted.
# 
# 4.**F1-Score**: The harmonic mean of precision and recall. It's a balance between them and is particularly useful if there's an imbalance in the class distribution.
# 
# 5.**ROC-AUC**: Area under the receiver operating characteristic curve. This tells you about the model's ability to discriminate between positive and negative classes.
# 
# *Based on these metrics, models that show a high area under the ROC curve (AUC), balanced precision and recall, and a higher F1-score would be considered better performers. For our models, Logistic Regression, Support Vector Machine, Random Forest, and Gradient Boosting have shown good performance. LightGBM and CatBoost also performed well, particularly CatBoost which exhibited a relatively balanced trade-off between precision and recall for the minority class.*
# 
# **Suggestions for Hyperparameter Tuning:**
# 
# **CatBoost**: It showed good balance in identifying both classes. It could benefit from hyperparameter tuning focused on learning rate, depth of trees, and l2_leaf_reg (regularization).
# 
# **LightGBM**: It also had strong performance and could see improvements by tuning the number of leaves, learning rate, and the number of trees.
# 
# **Gradient Boosting**: Already performing well but could see improvements in performance with tuning of learning rate, max_depth, and n_estimators.
# 
# **Random Forest**: Good performance could be enhanced by tuning the number of trees, max depth, and max features.

# ### Hyperparameter Tuning
# 
# **To select the best performing classifiers based on performance metrics for hyperparameter tuning, we can use GridSearchCV to search for the best hyperparameters for each classifier. We'll use the F1-score as the scoring metric because it balances precision and recall, which is important for binary classification problems like churn prediction.**

# In[25]:


# Define the parameter grid for each model
models_params = {
    'RandomForestClassifier': {
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [100, 300, 500],
        'classifier__max_features': ['auto', 'sqrt', 'log2'],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    },
    'GradientBoostingClassifier': {
        'classifier': [GradientBoostingClassifier()],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 10]
    },
    'LGBMClassifier': {
        'classifier': [LGBMClassifier()],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__num_leaves': [31, 62, 127],
        'classifier__boosting_type': ['gbdt', 'dart']
    },
    'CatBoostClassifier': {
        'classifier': [CatBoostClassifier(verbose=0)],
        'classifier__iterations': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__depth': [4, 6, 10]
    }
}

results = {}
for model_name, params in models_params.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', params['classifier'][0])])
    param_grid = {k: params[k] for k in params if k != 'classifier'}
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    results[model_name] = {'Best Parameters': best_params, 'Best Score': best_score}

for model, res in results.items():
    print(f"{model}: {res['Best Score']}, Parameters: {res['Best Parameters']}")


# **The hyperparameter tuning results for the selected models indicate the following best scores and parameters**:
# 
# **RandomForestClassifier** achieved the highest score of approximately 0.865, with optimal parameters being a maximum depth of 30, sqrt for max features, minimum samples split of 10, and 300 estimators.<br>
# **GradientBoostingClassifier** showed a score close to 0.863, with the best learning rate at 0.1, a max depth of 3, and 100 estimators.<br>
# **LGBMClassifier** also had a competitive score of about 0.863, with dart boosting type, a learning rate of 0.1, 200 estimators, and 31 as the optimal number of leaves.<br>
# **CatBoostClassifier** presented a score of 0.865, similar to RandomForest, with optimal depth of 4, 100 iterations, and a learning rate of 0.2.<br>
# 
# These results suggest that all four models perform similarly on this dataset, with slight variations in their optimal hyperparameters. RandomForest and CatBoost showed marginally better performance, indicating their effectiveness in handling this classification task. This similarity in scores highlights the importance of hyperparameter tuning in optimizing model performance.
