## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
  ### Developed by HARIPRASHAAD RA 
  ### Registered no: 212223040060
  ```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```

![image](https://github.com/user-attachments/assets/14c0344e-8231-4d9a-988b-5affbdf8f6c2)
```
`from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])

```
![image](https://github.com/user-attachments/assets/12543740-004a-467a-9171-165d8cdde29e)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/a27c04f5-250d-4d54-b91f-cc290aae2089)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/86b5b873-1702-43aa-b637-89d9cb99e389)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/7fece4e9-5051-4841-8173-469712fc1167)
```
pd.get_dummies(df2,columns=["nom_0"])

```
![image](https://github.com/user-attachments/assets/317d3e8c-ff99-4a8f-a923-24dd9a73d705)
```
pip install --upgrade category_encoders
```
```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
```
dfb=pd.concat([df,nd],axis=1)
dfb
```

![image](https://github.com/user-attachments/assets/97ab5473-5c21-476a-b60c-785903522331)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/1381a620-88b7-431f-bdd2-91cb35c37b34)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/25663380-68ee-491c-8777-5fee1e2583e5)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/c09f858e-5c95-4bb7-b2de-acf702237ffb)
```
np.log(df["Highly Positive Skew"])
```


![image](https://github.com/user-attachments/assets/1dfdba39-0c57-4c48-87b6-da3f4fed3503)
```
np.reciprocal(df["Moderate Positive Skew"])
```

![image](https://github.com/user-attachments/assets/31e19a04-fe60-4869-9979-65e4727b4a79)
```
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/48783199-6761-422a-a32e-b6ea7a8bfda4)

```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/6104f0c3-09bf-4fa0-bdbe-4a24a19cbec2)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![image](https://github.com/user-attachments/assets/42cb2da1-eb4d-4771-94ad-3721fe03ee32)

```
df.skew()
```

![image](https://github.com/user-attachments/assets/6d1c0ce9-90ee-4333-8b2c-1c41273347e6)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

![image](https://github.com/user-attachments/assets/f316f62b-29e1-45a5-815e-1fe661f8a840)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```


![image](https://github.com/user-attachments/assets/d71ed8a0-7c08-4566-b7ba-74aeb7479915)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/8a14f624-54b5-4dec-a856-c6eb28acc5b4)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/2ea95f9f-a4cc-436c-86a5-159a123e7e8a)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/452d7404-7fce-413f-b864-ab7ce20eddc1)


```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/a6597335-90c5-463f-86d2-7aea48eced58)
```
dt=pd.read_csv("titanic_dataset.csv")
dt
```
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```

![image](https://github.com/user-attachments/assets/39d1e28e-84ae-4e5e-9d20-5bf554bb8b04)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/8bb43756-6c9c-4e63-b197-4299a907cb90)





# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
