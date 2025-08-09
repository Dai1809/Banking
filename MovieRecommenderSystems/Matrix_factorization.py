import pandas as pd
import numpy as np

# Step 1: Load MovieLens 100K dataset
url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
columns = ['user_id', 'movie_id', 'rating', 'timestamp']
df = pd.read_csv(url, sep='\t', names=columns)
df.drop('timestamp', axis=1, inplace=True)
print(df.head())

# Step 2: Map original user/movie IDs to 0-based index
users = df['user_id'].unique()
movies = df['movie_id'].unique()

user_map = {id: i for i, id in enumerate(users)}
movie_map = {id: i for i, id in enumerate(movies)}

num_users = len(users)
num_movies = len(movies)

# Step 3: Create ratings matrix R
R = np.zeros((num_users, num_movies))

for row in df.itertuples(index=False):
    u = user_map[row.user_id]
    m = movie_map[row.movie_id]
    R[u, m] = row.rating

# Step 4: Matrix Factorization function
def matrix_factor(R, K=10, steps=100, alpha=0.01, lambda_=0.02):
    num_users, num_movies = R.shape
    U = np.random.normal(0, 0.1, (num_users, K))
    V = np.random.normal(0, 0.1, (num_movies, K))
    total_errors = []

    for step in range(steps):
        error = 0
        for u in range(num_users):
            for i in range(num_movies):
                if R[u, i] > 0:
                    pred = np.dot(U[u], V[i])
                    e = R[u, i] - pred
                    error += e**2

                    # Gradient Descent Updates
                    U[u] += alpha * (e * V[i] - lambda_ * U[u])
                    V[i] += alpha * (e * U[u] - lambda_ * V[i])

        total_errors.append(error)
        if step % 10 == 0:
            rmse = np.sqrt(error / np.count_nonzero(R))
            print(f"Step {step}, RMSE: {rmse:.4f}")

    return U, V, total_errors

# Step 5: Train the model
U, V, total_errors = matrix_factor(R, K=10, steps=100, alpha=0.01, lambda_=0.02)
predicted_R = np.dot(U, V.T)

# Step 6: Recommend top 5 movies for user 0
user_index = 0
user_ratings = predicted_R[user_index]
already_rated = R[user_index] > 0

# Get top 5 unrated movie indices
unrated_indices = np.where(~already_rated)[0]
top_unrated = np.argsort(user_ratings[unrated_indices])[-5:]
recommend_indices = unrated_indices[top_unrated]

print("Top 5 recommended movie indices for user 0:")
for idx in recommend_indices:
    print(f"Movie Index: {idx}, Predicted Rating: {user_ratings[idx]:.2f}")
