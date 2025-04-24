
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re
from sklearn.metrics.pairwise import cosine_similarity

# Set random seed for reproducibility
np.random.seed(42)

# --- Step 1: Fetch Real Recipe Data ---
def fetch_recipe_data():
    url = "https://starfish-app-9uavi.ondigitalocean.app/recipes"
    try:
        response = requests.get(url)
        data = response.json()
        recipes = data.get('recipes', [])
        return recipes
    except Exception as e:
        print(f"Error fetching recipe data: {e}")
        return []

# Fetch recipes from API
recipes_data = fetch_recipe_data()

# Process recipes data into dataframe format
def process_recipes(recipes):
    processed_data = []
    
    for recipe in recipes:
        # Extract cooking time in minutes
        time_str = recipe.get('Time', '0 minutes')
        cooking_time = 0
        
        # Parse time string to get minutes
        if 'hour' in time_str.lower() and 'minute' in time_str.lower():
            # Format like "1 hour 30 minutes"
            time_parts = time_str.split()
            hours = int(time_parts[0])
            minutes = int(time_parts[2])
            cooking_time = hours * 60 + minutes
        elif 'hour' in time_str.lower():
            # Format like "2 hours"
            hours = int(re.search(r'(\d+)', time_str).group(1))
            cooking_time = hours * 60
        elif 'minute' in time_str.lower():
            # Format like "45 minutes"
            minutes = int(re.search(r'(\d+)', time_str).group(1))
            cooking_time = minutes
        
        # Determine difficulty based on cooking time and number of ingredients
        ingredient_count = len(recipe.get('ingredients', []))
        if cooking_time < 30 and ingredient_count < 7:
            difficulty = 1  # Easy
        elif cooking_time < 60 and ingredient_count < 12:
            difficulty = 2  # Medium
        else:
            difficulty = 3  # Hard
        
        # Simulate engagement metrics based on recipe attributes
        views = np.random.randint(5000, 50000)
        engagement_factor = (5 - abs(recipe.get('calories', 500) - 600) / 200) / 5  # Preference for moderate calories
        quality_factor = min(5, recipe.get('rating', 4)) / 5  # Use recipe rating if available
        
        likes = int(views * engagement_factor * quality_factor * np.random.uniform(0.05, 0.15))
        dislikes = int(likes * np.random.uniform(0.01, 0.1))
        shares = int(likes * np.random.uniform(0.1, 0.3))
        comments = int(likes * np.random.uniform(0.2, 0.5))
        
        # Calculate average watch time (based on cooking time and recipe quality)
        watch_percentage = 0.6 + 0.3 * quality_factor + 0.1 * np.random.random()
        watch_time = int(cooking_time * 60 * watch_percentage)  # In seconds
        
        # Simulated trending factor - some recipes are trending up, others down
        trend_factor = np.random.uniform(0.5, 1.5)
        
        processed_data.append({
            'recipe_id': recipe.get('id'),
            'title': recipe.get('name'),
            'category': recipe.get('category', 'undefined'),
            'author': recipe.get('author', 'unknown'),
            'calories': recipe.get('calories', 0),
            'carbs': recipe.get('carbs', 0),
            'fats': recipe.get('fats', 0),
            'proteins': recipe.get('proteins', 0),
            'cooking_time': cooking_time,
            'difficulty': difficulty,
            'servings': recipe.get('servings', 1),
            'views': views,
            'likes': likes,
            'dislikes': dislikes,
            'shares': shares, 
            'comments': comments,
            'watch_time': watch_time,
            'rating': recipe.get('rating', np.random.uniform(3.0, 5.0)),
            'trend_factor': trend_factor,
            'ingredient_count': ingredient_count
        })
    
    return pd.DataFrame(processed_data)

# Create DataFrame with processed recipe data
df = process_recipes(recipes_data)

# Infer categories if missing (based on title keywords)
def infer_category(title):
    title = title.lower()
    if any(keyword in title for keyword in ['pizza', 'pasta', 'steak', 'chicken', 'beef', 'pork', 'fish']):
        return 'main dish'
    elif any(keyword in title for keyword in ['cake', 'cookie', 'dessert', 'sweet', 'pie', 'chocolate', 'ice cream', 'cinnamon']):
        return 'dessert'
    elif any(keyword in title for keyword in ['soup', 'salad', 'appetizer', 'dip', 'snack']):
        return 'appetizer'
    elif any(keyword in title for keyword in ['breakfast', 'pancake', 'waffle', 'egg', 'muffin']):
        return 'breakfast'
    else:
        return 'other'

# Apply category inference where needed
df['category'] = df.apply(lambda row: row['category'] if row['category'] != 'undefined' else infer_category(row['title']), axis=1)

# --- Step 2: Feature Engineering ---
# Basic engagement metrics
df['like_ratio'] = df['likes'] / (df['likes'] + df['dislikes'] + 1)  # Add 1 to avoid division by zero
df['engagement'] = df['likes'] + df['dislikes']
df['completion_rate'] = df['watch_time'] / (df['cooking_time'] * 60 + 1)  # Add 1 to avoid division by zero
df['nutrient_balance'] = np.exp(-(np.abs(df['carbs'] - 50) + np.abs(df['fats'] - 25) + np.abs(df['proteins'] - 25))/100)
df['ingredient_complexity'] = df['ingredient_count'] / 10  # Normalized ingredient count
df['social_score'] = df['shares'] + df['comments']  # Combined social metrics
df['engagement_factor'] = (df['likes'] + df['comments'] + 3*df['shares']) / (df['views'] + 1)  # Engagement as percentage of views

# --- Step 3: Eigenvalue-Based Ranking Algorithm ---
# Normalize the feature data for better comparison
scaler = StandardScaler()
features = [
    'views', 'likes', 'rating', 'social_score', 
    'engagement_factor', 'like_ratio', 'completion_rate', 
    'difficulty', 'cooking_time'
]

# Create a copy of the feature data for scaling
feature_data = df[features].copy()
feature_data_scaled = scaler.fit_transform(feature_data)

# Get number of recipes
n = len(df)

# Calculate similarity matrix using cosine similarity
similarity_matrix = cosine_similarity(feature_data_scaled)

# Create recipe transition matrix (similar to how PageRank works)
# Higher engagement and quality means more likely to transition to that recipe
engagement_influence = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:  # Don't compare recipe to itself
            # Calculate how much recipe j influences recipe i
            # based on engagement and similarity
            engagement_influence[i, j] = (
                df.iloc[j]['likes'] / df['likes'].mean() *
                df.iloc[j]['views'] / df['views'].mean() *
                df.iloc[j]['rating'] / df['rating'].mean() *
                max(0.1, similarity_matrix[i, j])  # Ensure non-zero probability
            )

# Normalize rows to create proper transition probabilities
row_sums = engagement_influence.sum(axis=1)
transition_matrix = engagement_influence / row_sums[:, np.newaxis]

# Add damping factor (like in PageRank algorithm)
damping = 0.85
random_surf_prob = (1 - damping) / n
transition_matrix = damping * transition_matrix + random_surf_prob

# Power iteration method to find the principal eigenvector
# (simpler than full eigendecomposition for this purpose)
v = np.ones(n) / n  # Initial uniform probability
tolerance = 1e-8
max_iterations = 100
for _ in range(max_iterations):
    v_prev = v
    v = np.dot(transition_matrix.T, v)
    # Check for convergence
    if np.linalg.norm(v - v_prev) < tolerance:
        break

# Normalize the eigenvector to get scores between 0 and 1
df['eigenrank_score'] = v / max(v)

# Sort recipes by eigenvalue score
df = df.sort_values('eigenrank_score', ascending=False).reset_index(drop=True)

# Store component scores for display
df['quality_score'] = df['rating'] / 5.0
df['engagement_score'] = df['engagement_factor']

# --- Step 4: Output Top 20 Ranked Recipes ---
print("\n" + "="*80)
print("🏆 TOP 20 RANKED RECIPES (EIGENVALUE RANKING) 🏆".center(80))
print("="*80)

for i, row in df.head(20).iterrows():
    print(f"{i+1}. {row['title']} ({row['category']}) - Score: {row['eigenrank_score']:.2f}")
    print(f"   RATING: ★{row['rating']:.1f}/5.0 | LIKES: {int(row['likes']):,} | VIEWS: {int(row['views']):,}")
    print(f"   Engagement Factor: {row['engagement_factor']*100:.1f}% | Social: {int(row['social_score']):,}")
    print(f"   Author: {row['author']} | Quality: {row['quality_score']:.2f} | Engagement: {row['engagement_score']:.2f}")
    print()

# Save results
df.to_csv('ranked_recipes_eigenvalue.csv', index=False)
print("\nTop 20 recipes saved to ranked_recipes_eigenvalue.csv")
