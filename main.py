import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import requests
import streamlit as st

def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# feature engineering
# pd.set_option('future.no_silent_downcasting', True)
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


# Load the dataset
file_path = 'indian_food.csv' 
df = pd.read_csv(file_path)

st.image("dishcoverlogo.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

# Unique course types (appetizer, main course, dessert, etc.)
courses = df['course'].unique()

# Separate selection for each course
for course in courses:
    st.title(f"{course.capitalize()}")
    selected_dishes = st.multiselect(
        f"Select {course} dishes", 
        df[df['course'] == course]['name'].unique()
    )

    # Display selected dishes with fetched images
    if selected_dishes:
        selected_df = df[df['name'].isin(selected_dishes)]
        for dish in selected_dishes:
            inputidx = df.index.get_loc(df[df['name'] == dish].index[0])
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

            x = 3  # Number of recommendations
            topx = cosine_similarities[1 : x + 1]  # Skip the first one, which is the input itself
            # Output
            cosine_values = np.array([x[0] for x in topx])
            softmax_probs = softmax(cosine_values)
            match_percentages = softmax_probs * 100
            target_percentage = 20.0
            match_scores = 100 - np.abs(match_percentages - target_percentage)
            match_scores = np.clip(match_scores, 0, None)
            
            
            # Output
            st.header(f"Previously enjoyed: {names.iloc[inputidx]}")
            st.subheader(f"Recommended dishes: ") 
            for i in range(x):
                st.markdown(f'{names.iloc[topx[i][1]]} - {match_scores[i]:.2f}%')
            st.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
                list-style-position: inside;
            }
            </style>
            ''', unsafe_allow_html=True)

print(selected_dishes)


