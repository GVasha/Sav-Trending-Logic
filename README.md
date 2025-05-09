﻿# Sav-Trending-Logic
 Absolutely! Let's walk through your code step-by-step, explaining each major part clearly. Your code is a great example of **applied data science with linear algebra concepts**, especially around feature scaling and scoring systems.

---

## 🔢 **Overview:**
The code ranks recipes from your "Savory" API by simulating metrics (likes, views, shares, etc.) and combining them using a weighted scoring system.

---

## 📌 **Step-by-Step Explanation**

---

### **Step 1: Fetch Real Recipe Data**
```python
def fetch_recipe_data():
    ...
```
- This function uses the `requests` module to **pull real recipe data from your backend API**.
- Returns a list of recipes, each with attributes like `calories`, `ingredients`, etc.

---

### **Step 2: Process Recipes**
```python
def process_recipes(recipes):
    ...
```
- Converts raw recipe data into a **structured DataFrame**.
- Adds simulated fields:
  - `views`, `likes`, `dislikes`, `comments`, `shares`, `watch_time`
- **Cooking Time Parsing**: Converts text like "1 hour 30 minutes" to `90`.
- **Difficulty Score**: Based on cooking time and number of ingredients.

#### 🔧 Simulation logic:
- Engagement metrics are **randomly generated but based on logic**:
  - **More balanced calorie values = better**.
  - **Higher rating = better quality factor**.
  - These affect how many likes, comments, etc. a recipe gets.

---

### **Step 3: Infer Missing Categories**
```python
def infer_category(title):
    ...
```
- If a recipe category is `'undefined'`, the function guesses it based on keywords in the title (e.g., "cake" → dessert).

---

### **Step 4: Feature Engineering**
You create new columns to better understand engagement and quality:
```python
df['like_ratio'] = ...
df['completion_rate'] = ...
df['nutrient_balance'] = ...
```
#### 🔬 What's going on:
- **like_ratio**: Likes vs. total reactions.
- **completion_rate**: How much of the video people watch.
- **nutrient_balance**: Penalizes overly high/low carbs, fats, proteins.
- **normalized_likes/views**: `log1p()` helps normalize skewed distributions (like exponential views).

These are your **feature vectors**: numerical values that represent each recipe.

---

### **Step 5: Normalize Feature Groups**
```python
scaler = MinMaxScaler()
...
engagement_scaled = scaler.fit_transform(...)
...
```
- You normalize features so they're all between 0 and 1.
- This is a **linear transformation** using the `MinMaxScaler`.
- Divides features into:
  - **Engagement**
  - **Quality**
  - **Complexity**
  - **Trend**

---

### **Step 6: Compute Component Scores**
```python
df['engagement_score'] = np.average(engagement_scaled, axis=1, weights=[...])
...
```

- Here you compute **weighted averages** of the normalized features.
- This is a **dot product**: feature vector ⋅ weight vector.
- More weight is given to important features (e.g., views, likes, rating).

---

### **Step 7: Final Score**
```python
df['final_score'] = (
    score_weights['engagement'] * df['engagement_score'] +
    ...
)
```
- Combines all 4 scores into a **single ranking score**.
- **Engagement** and **quality** get much more weight than trend or accessibility.

---

### **Step 8: Display Results**
```python
for i, row in df.head(20).iterrows():
    print(...)
```
- Loops through the top 20 recipes and prints their scores and stats in a nice format.

---

### **Step 9: Save to CSV**
```python
df.to_csv('ranked_recipes.csv', index=False)
```
- Exports your ranked recipe table for further use or visualization.

---

## ✅ **Key Linear Algebra Concepts in Use**

| Concept | Where It Appears |
|--------|------------------|
| **Vectors** | Recipe features like `[likes, views, rating, carbs, etc.]` |
| **Matrices** | All recipe feature vectors together = a feature matrix (n × d) |
| **Linear Transformation** | `MinMaxScaler` normalizes features linearly |
| **Dot Product / Weighted Sum** | Scoring formulas use weights and `np.average` |
| **Feature Engineering** | Transforms raw data into numerical vector form |
| *(Optional)* **SVD/PCA** | You could later use this to reduce feature space |

---
