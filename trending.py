import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Fetch data
url = "https://starfish-app-9uavi.ondigitalocean.app/recipes"
response = requests.get(url)
data = response.json()

# Step 2: Parse and enrich data
records = []
for r in data["recipes"]:
    ingredients = r.get("ingredients", [])
    num_ingredients = len(ingredients)
    quantified_ingredients = [i for i in ingredients if i.get("quantity") is not None]
    num_quantified = len(quantified_ingredients)
    avg_quantity = np.mean([i["quantity"] for i in quantified_ingredients]) if quantified_ingredients else 0
    
    # Extract time in minutes
    try:
        time_str = r.get("Time", "0").lower().replace("minutes", "").replace("minute", "").strip()
        cooking_time = int(''.join(filter(str.isdigit, time_str))) if time_str else 0
    except:
        cooking_time = 0

    servings = r.get("servings", 1)
    calories = r.get("calories", 0)
    
    records.append({
        "id": r["id"],
        "name": r["name"],
        "calories": calories,
        "carbs": r.get("carbs", 0),
        "fats": r.get("fats", 0),
        "proteins": r.get("proteins", 0),
        "servings": servings,
        "calories_per_serving": calories / servings if servings > 0 else calories,
        "cooking_time": cooking_time,
        "num_ingredients": num_ingredients,
        "quantified_ingredients": num_quantified,
        "avg_quantity": avg_quantity,
        "is_vegan": int(r.get("is_vegan", False)),
        "is_vegetarian": int(r.get("is_vegetarian", False)),
        "is_gluten_free": int(r.get("is_gluten_free", False)),
        "is_lactose_free": int(r.get("is_lactose_free", False)),
    })

df = pd.DataFrame(records)

# Step 3: Feature engineering
df["inv_calories_per_serving"] = -df["calories_per_serving"]
df["inv_carbs"] = -df["carbs"]
df["inv_fats"] = -df["fats"]
df["high_protein"] = df["proteins"]
df["inv_cooking_time"] = -df["cooking_time"]
df["inv_ingredients"] = -df["num_ingredients"]
df["diverse_ingredients"] = df["quantified_ingredients"]
df["avg_quantity"] = df["avg_quantity"]

# Step 4: Select and scale features
features = [
    "inv_calories_per_serving", "inv_carbs", "inv_fats",
    "high_protein", "inv_ingredients", "inv_cooking_time",
    "diverse_ingredients", "avg_quantity",
    "is_vegan", "is_vegetarian", "is_gluten_free", "is_lactose_free"
]
X = df[features]
X_scaled = StandardScaler().fit_transform(X)

# Step 5: SVD
U, S, VT = np.linalg.svd(X_scaled, full_matrices=False)
df["svd_score"] = U[:, 0] * S[0]
df["svd_2"] = U[:, 1] * S[1]

# Step 6: PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)
df["pca_score"] = pca_components[:, 0]
df["pca_2"] = pca_components[:, 1]

# Step 7: Top 10 by SVD
top_svd = df.sort_values("svd_score", ascending=False).head(10)
plt.figure(figsize=(12, 5))
sns.barplot(data=top_svd, x="svd_score", y="name", palette="Blues_d")
plt.title("Top 10 Healthiest Recipes by Enhanced SVD Score")
plt.xlabel("SVD-Based Health Score")
plt.ylabel("Recipe Name")
plt.tight_layout()
plt.show()

# Step 8: Top 10 by PCA
top_pca = df.sort_values("pca_score", ascending=False).head(10)
plt.figure(figsize=(12, 5))
sns.barplot(data=top_pca, x="pca_score", y="name", palette="Greens_d")
plt.title("Top 10 Healthiest Recipes by PCA Score")
plt.xlabel("PCA-Based Health Score (PC1)")
plt.ylabel("Recipe Name")
plt.tight_layout()
plt.show()

# Step 9: Scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x="svd_score", y="pca_score", hue="is_vegan", palette="Set2", s=100)
plt.title("SVD vs PCA Health Score Comparison (Enhanced Features)")
plt.xlabel("SVD-Based Score")
plt.ylabel("PCA-Based Score (PC1)")
plt.axhline(0, linestyle='--', color='gray')
plt.axvline(0, linestyle='--', color='gray')
plt.legend(title="Vegan")
plt.tight_layout()
plt.show()
