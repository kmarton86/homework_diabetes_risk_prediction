from sklearn.datasets import load_diabetes
import pandas as pd

diabetes = load_diabetes()

print(type(diabetes))
print(diabetes.data.shape)
# (442, 10) → 442 sor, 10 feature
print(diabetes.feature_names)

# ----------------------
# Convert to DataFrame
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df["target"] = diabetes.target

# print(df.head())
# print(df.columns)
print(df)