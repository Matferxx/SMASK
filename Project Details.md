**Intro and Imports**

```
import os
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

# Check if files are there

```
```
import os
os.chdir('/content/drive/My Drive/SMASK_Project')
!ls
```
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

Based on this we conclude

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

-------------------------------------------------------------------------------------




