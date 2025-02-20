**Done with inspiration from:**
https://github.com/piyushg9794/Bike-Rental-Demand-forecasting/blob/master/model.ipynb

**Link to collab**
https://colab.research.google.com/drive/1fvFFRs-Iy0eo3cpYASiT-6iCCrSMo1vD?usp=sharing







**Intro and Imports**

```

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.tree as skl_tree
import sklearn.ensemble as skl_en
import sklearn.svm as skl_svm
import sklearn.neural_network as skl_nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from IPython.core.pylabtools import figsize
from sklearn.model_selection import train_test_split

```
```

url = "https://raw.githubusercontent.com/Matferxx/SMASK/main/training_data_vt2025.csv"
df = pd.read_csv(url)
```
-------------------------------------------------------------------------------------
**Data Info**
```
df.describe()
df.info()
df
```

![image](https://github.com/user-attachments/assets/990844a8-a2d4-48b5-b5fd-534770ad27eb)




-------------------------------------------------------------------------------------
**Plots and Data analysis**

```
import plotly.express as px

# Select numerical features
numerical_features = ['temp', 'dew', 'humidity', 'visibility', 'windspeed', 'precip']

# Create scatter matrix plot
fig = px.scatter_matrix(df, dimensions=numerical_features, color="increase_stock",
                        color_discrete_map={0: "blue", 1: "red"},  # 0 = low demand, 1 = high demand
                        labels={"increase_stock": "Bike Demand"})

# Show plot
fig.show()

```

![newplot](https://github.com/user-attachments/assets/cbc24239-9b56-4b15-b8a5-270bff8d8dfa)

````

# Compute the correlation matrix
correlation_matrix = df.corr()

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Features", fontsize=14, fontweight="bold")
plt.show()

````

![image](https://github.com/user-attachments/assets/28f5144b-6011-4d68-ae54-06a2f2e2e772)



-------------------------------------------------------------------------------------
**Pre-Processing - From here and below, the processing of the data is done**


Checks nulls:
```
df.isnull().sum
```
Drops snow

```
columnsToDrop= ['snow']
df.drop(columnsToDrop, axis=1, inplace=True)
```
Checks missing values and replaces with median

```
# Check for missing values
print(df.isnull().sum())

# Fill numerical missing values with median
df.fillna(df.median(), inplace=True)

# Fill categorical missing values with mode
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing values handled.")

```
-------------------------------------------------------------------------------------

**Normalizing and scaling**

**Z-score** standardization (which assumes normality).

**Log transformation** helps compress large values and spreads smaller values for a more normal-like distribution.

**Min-Max scaling** is useful if you want to keep the range between [0,1] without changing the shape much.

visualizing distribution of data to analyze skewness & presence of outliers in order to determine the distribution of the different features
```
columns = df.columns.values
for x in columns:
    sns.distplot(df[x], kde=True)
    plt.show()
```
![Increase_Stock](https://github.com/user-attachments/assets/01ebcf25-8a8a-413f-b673-bbdd244a14e9)
![Visibility](https://github.com/user-attachments/assets/7a06f162-90f7-4332-bf74-c05d15c2204e)
![Cloudcover](https://github.com/user-attachments/assets/dbbcfee1-1a1a-4fab-965b-914180201eeb)
![Windspeed](https://github.com/user-attachments/assets/4e82d252-d751-4964-9087-2e0f2dca1635)
![Humidity](https://github.com/user-attachments/assets/c8856a8b-216a-4865-8bcf-e8fd116effe1)
![Dew](https://github.com/user-attachments/assets/7fca23c0-d384-4a18-bf8a-88a424657d7b)
![Temp](https://github.com/user-attachments/assets/f4ec8af6-e35e-4061-b823-b6b350bf705d)
![Summertime](https://github.com/user-attachments/assets/45f5bf1c-1070-412d-804d-542b6b83d2a0)
![Weekday](https://github.com/user-attachments/assets/d4f9dbd1-2e0f-4b17-a662-661c34ab59d4)
![Holiday](https://github.com/user-attachments/assets/307a7797-6ef7-4989-81eb-d90f471a0919)
![Month](https://github.com/user-attachments/assets/a20e753a-99f7-4e31-81c2-e1161d8b4344)
![Day_of_week](https://github.com/user-attachments/assets/d14085bf-276c-4f4d-bd36-b4ab77015aaf)
![Hour_of_day](https://github.com/user-attachments/assets/59a1db10-a564-40cc-b3b1-e9139bcd6e71)

Based on these graphs, I conclude that the preceding handling of the data will be:

**hour_of_day	Numerical (Normal)**	Z-score (Standardization),

**day_of_week	Categorical (Cyclical)**	Sine-Cosine Encoding or One-Hot
month	Categorical	One-Hot Encoding

**holiday	Binary Categorical**	No Normalization (Already 0/1)

**weekday	Binary Categorical**	No Normalization (Already 0/1)

**summertime	Binary Categorical**	No Normalization (Already 0/1)

**temp	Numerical (Normal)**	Z-score (Standardization)

**dew	Numerical (Normal)**	Z-score (Standardization)

**humidity	Numerical (Normal)**	Z-score (Standardization)

**windspeed	Numerical (Normal)**	Z-score (Standardization)

**cloudcover	Numerical (Skewed)**	Min-Max Scaling

**visibility	Numerical (Skewed)**	Min-Max Scaling

**increase_stock**	Target Variable	No Normalization (Already Binary 0/1)




**Scales**
```

# Initialize scalers
z_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

# Apply Z-score Standardization (for normally distributed numerical features)
zscore_features = ['hour_of_day', 'temp', 'dew', 'humidity', 'windspeed']
df[zscore_features] = z_scaler.fit_transform(df[zscore_features])

# Apply Min-Max Scaling (for skewed numerical features)
minmax_features = ['cloudcover', 'visibility']
df[minmax_features] = minmax_scaler.fit_transform(df[minmax_features])

# Encode Categorical Features
df = pd.get_dummies(df, columns=['month'], prefix='month')  # One-hot encode month
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)  # Sine encoding for weekdays
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)  # Cosine encoding for weekdays
df.drop(columns=['day_of_week'], inplace=True)  # Remove original categorical column

# Target variable remains as is (Binary 0/1)
print("Feature transformation complete!")

```

**Checks scaling**
```

# Check summary statistics of transformed features
print("Summary statistics of transformed features:\n")
print(df.describe())

# Visualize Z-score Standardized Features
zscore_features = ['hour_of_day', 'temp', 'dew', 'humidity', 'windspeed']
plt.figure(figsize=(12, 6))
for i, feature in enumerate(zscore_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[feature], kde=True)
    plt.title(f"Standardized {feature}")
plt.tight_layout()
plt.show()

# Visualize Min-Max Scaled Features
minmax_features = ['cloudcover', 'visibility']
plt.figure(figsize=(10, 4))
for i, feature in enumerate(minmax_features, 1):
    plt.subplot(1, 2, i)
    sns.histplot(df[feature], kde=True)
    plt.title(f"Min-Max Scaled {feature}")
plt.tight_layout()
plt.show()

print("Feature verification complete!")
```
**Creating data split using seed so the data is uniform over models**

```

#Creating data split 

from sklearn.model_selection import train_test_split

# Define feature matrix (X) and target variable (y)
X = df.drop(columns=['increase_stock'])  # Drop target column
y = df['increase_stock']  # Target variable

# Split data into training (80%) and testing (20%) sets with a fixed random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Verify shape
print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

```



-------------------------------------------------------------------------------------






