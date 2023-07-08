import pandas as pd
import numpy as np

dataset = pd.read_csv("50_Startups.csv")
print(dataset)

# .shape gives us the number of rows and columns
dataset.shape

#
x = dataset[["R&D Spend", "Administration", "Marketing Spend", "State"]]
x

y = dataset[["Profit"]]
y


from sklearn.impute import SimpleImputer

dataset.replace(0.00, np.nan, inplace=True)
print(dataset)
my_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
dataset[["R&D Spend", "Marketing Spend"]] = my_imputer.fit_transform(
    dataset[["R&D Spend", "Marketing Spend"]]
).astype("object")
dataset


x = dataset[["R&D Spend", "Administration", "Marketing Spend", "State"]].values
y = dataset[["Profit"]].values
print(x)
print(y)


from sklearn.preprocessing import LabelEncoder

label_encode_x = LabelEncoder()
x[:, 3] = label_encode_x.fit_transform(x[:, 3])
print(x)

from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder()

onehotencoder.fit_transform(dataset.State.values.reshape(-1, 1)).toarray()
print(onehotencoder.fit_transform(dataset.State.values.reshape(-1, 1)).toarray())


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

print(x_train)
print(x_test)
print(y_train)
print(y_test)


from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

print(x_train)

print(x_test)
