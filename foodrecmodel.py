import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# feature engineering

df = pd.read_csv("indian_food.csv")
df_dummies = pd.get_dummies(df, columns=["region"], drop_first=True)
df1 = df_dummies.replace({True: 1, False: 0})
df2 = df1.drop(columns=["prep_time", "cook_time"])
df2["diet_encoded"] = df2["diet"].apply(lambda x: 0 if x == "vegetarian" else 1)
df2 = df2.drop(columns=["diet"])
df3 = pd.get_dummies(df2, columns=["flavor_profile", "course"], drop_first=True)
df3 = df3.replace({True: 1, False: 0})

mlb = MultiLabelBinarizer()
# One-hot encode the ingredients
ingredients_onehot = mlb.fit_transform(df3["ingredients"])
# Convert the result to a DataFrame
ingredients_df = pd.DataFrame(ingredients_onehot, columns=mlb.classes_)
# Concatenate the one-hot encoded DataFrame with the original DataFrame
df4 = pd.concat([df3, ingredients_df], axis=1)

# Fix 'yogurt' and 'yoghurt'
if "yoghurt" in df4.columns and "yogurt" in df4.columns:
    df4["yogurt"] += df4["yoghurt"]
    df4 = df4.drop(columns=["yoghurt"])

df4 = df4.drop(columns=["ingredients"])
df4 = pd.get_dummies(df4, columns=["state"], prefix=["state"], dtype=int)

# Saving 'name' for recommendations later
names = df3["name"].copy()

"""
The input index can be adjusted to take
name input.

# Get the index based on the dish name input
dish_name = input("Enter the name of the dish: ")
inputidx = names[names == dish_name].index[0]  # Find the first match

Assuming the user inputs the exact name of input

"""

inputidx = 110  # pick some arbitrary index
cosine_similarities = []
df4_numerical = df4.drop(columns=["name"])  # Drop 'name' only for numerical purposes
df4_numerical = df4_numerical.astype(float)

v1 = df4_numerical.iloc[inputidx]

# Compute cosine similarities
for i in range(len(df4_numerical)):
    v2 = df4_numerical.iloc[i]
    cosine_similarities.append(
        (np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), i)
    )

# Sort the similarities
cosine_similarities.sort(reverse=True, key=lambda x: x[0])

x = 5  # Number of recommendations
topx = cosine_similarities[1 : x + 1]  # Skip the first one, which is the input itself

# Output
print(f"Previously enjoyed: {names.iloc[inputidx]}")
print(f"Recommended dishes: ")
for i in range(x):
    print(names.iloc[topx[i][1]])
