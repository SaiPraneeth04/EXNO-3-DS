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
1. FUNCTION TRANSFORMATION:
     
    • Log Transformation
   
    • Reciprocal Transformation
   
    • Square Root Transformation
   
    • Square Transformation

3. POWER TRANSFORMATION:
  
    • Boxcox method
   
    • Yeojohnson method

# CODING AND OUTPUT:

```
/* DEVELOPED BY :Sai Praneeth K
REGISTER NO: 212222230067*/
```
```
import pandas as pd
df = pd.read_csv("Encoding_data.csv")
df.head()
```
![1](https://github.com/user-attachments/assets/156fc3c4-f5f5-41f1-b488-80e4563be125)


```
df.tail()
```
![2](https://github.com/user-attachments/assets/1abaa5ec-bd48-4342-a8ac-b13a08ec0019)


```
df.describe()
```
![3](https://github.com/user-attachments/assets/d976df12-387c-493e-bfe2-73286cdbbf46)


```
df.info()
```
![4](https://github.com/user-attachments/assets/706d0152-7feb-4f6e-993f-2fc8bbe624a1)


```
df.shape
```
![5](https://github.com/user-attachments/assets/a66f3154-4473-4b36-89dc-ce7c5b643793)


```
df
```
![6](https://github.com/user-attachments/assets/389cd8b4-1b40-4abd-b9a0-d5a59d91b3b2)


```
#ordinal encoder
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
pm=['Hot', 'Warm','Cold']
oe=OrdinalEncoder(categories=[pm])
oe.fit_transform(df[["ord_2"]])
```
![7](https://github.com/user-attachments/assets/793f141e-75fb-40d0-9718-d580d8f86535)


```
df['bo2']=oe.fit_transform(df[["ord_2"]])
df
```
![8](https://github.com/user-attachments/assets/6460cf51-6a44-4560-8f6f-4742c777f415)


```
#label Encoder
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![9](https://github.com/user-attachments/assets/ecc7262d-dff5-4666-a3a3-87f507c26bcb)


```
#One hot encoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output = False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![10](https://github.com/user-attachments/assets/5e32a91d-e4fe-460c-a3bc-21c894d91649)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![11](https://github.com/user-attachments/assets/c89d892d-dd07-4024-be55-133795a999e4)


```
pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
df= pd.read_csv("data.csv")
df
```
![12](https://github.com/user-attachments/assets/7386505d-9454-40b8-9fc8-672b007dc46c)

```
#binary encoder
be = BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb

```
![13](https://github.com/user-attachments/assets/fc42ab4e-9629-45ff-a3e7-91159e24b057)


```
#target encoder
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new = te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![14](https://github.com/user-attachments/assets/f20690b9-082d-4e0b-ab1f-739d9f3351d2)


```
#Feature Transformation
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![15](https://github.com/user-attachments/assets/53bd70bb-541b-453f-87d2-a253d2c9c682)


```
df.info()
```
![16](https://github.com/user-attachments/assets/df51c000-1190-4601-a6c5-d5b8636674fc)


```
df.describe()
```
![17](https://github.com/user-attachments/assets/50e30097-8d39-472d-80be-0a1b3a839802)


```
df.size
```
![18](https://github.com/user-attachments/assets/2290e080-e5be-4ac3-8985-d22c84386db6)

```
df.skew()
```
![19](https://github.com/user-attachments/assets/7500ac87-d766-4b5a-9b0b-16b435aad362)


```
np.log(df["Highly Positive Skew"])
```
![20](https://github.com/user-attachments/assets/276fc65c-ede4-4fed-8c25-a9a07977af56)


```
np.reciprocal(df["Moderate Positive Skew"])
```
![21](https://github.com/user-attachments/assets/c2e2ad52-0d60-4696-9b9f-33b5a788cde4)


```
np.sqrt(df["Highly Positive Skew"])
```
![22](https://github.com/user-attachments/assets/2c786043-0afc-48db-99a0-7f0078cd2d6b)

```
np.square(df["Highly Positive Skew"])
```
![23](https://github.com/user-attachments/assets/395c0986-0233-4498-b351-646005757286)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![24](https://github.com/user-attachments/assets/4a6a136e-c430-40fc-a39d-df52af1b9a06)


```
df["Moderate Negative Skew_yeojohnson"],parameters =stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![25](https://github.com/user-attachments/assets/c05e0738-7e44-4c36-9dd7-c6e8d6645a28)


```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![26](https://github.com/user-attachments/assets/d65bd856-bef0-4316-8352-f76a4a84f811)


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])


sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![27](https://github.com/user-attachments/assets/7b1c3777-f632-45d8-bd2e-297f30ac48cf)


```
df
```
![28](https://github.com/user-attachments/assets/5b0c6aaa-ee59-4fe6-ab82-ea8e18ec9776)


```
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![29](https://github.com/user-attachments/assets/e42b69e1-6bc5-4721-9e73-05bd1d9136bd)



```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![30](https://github.com/user-attachments/assets/eb5d458b-3e49-4b41-bb4e-eb94a0fd71d4)


```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![31](https://github.com/user-attachments/assets/dadfae09-88b1-4b16-9612-f4cd99cbb8c2)



```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![32](https://github.com/user-attachments/assets/28608fa9-7510-4efe-9058-8ddf20c52411)



# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
