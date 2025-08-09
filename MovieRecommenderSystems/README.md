# 🎯 Movie Recommender Systems (ML-100K Dataset)

This project implements **four different recommendation system approaches** using the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/).  
It demonstrates **Collaborative Filtering** (User-Based & Item-Based), **Matrix Factorization**, and **Content-Based Filtering**.

---

## 📂 Dataset
We use the **MovieLens 100K dataset**:
- **Ratings file** (`u.data`): Contains `user_id`, `movie_id`, `rating`
- **Movies file** (`u.item`): Contains `movie_id`, `title`, and genre flags
- 100,000 ratings from 943 users on 1,682 movies

---

## 🧠 Implemented Methods

### 1️⃣ User-Based Collaborative Filtering
- Finds **similar users** based on their movie rating patterns.
- Predicts a user's rating for a movie by looking at ratings from **nearest neighbors** (users with similar tastes).

**Key Steps:**
- Create a **user-movie ratings matrix**
- Compute **cosine similarity** between users
- Recommend movies rated highly by similar users that the target user hasn’t seen

---

### 2️⃣ Item-Based Collaborative Filtering
- Finds **similar movies** instead of similar users.
- Predicts ratings by finding movies similar to the ones a user has already liked.

**Key Steps:**
- Create a **movie-user matrix**
- Compute **cosine similarity** between items
- Recommend movies similar to those the user has rated highly

---

### 3️⃣ Matrix Factorization (with Gradient Descent)
- Decomposes the ratings matrix `R` into **two low-dimensional matrices**:
  - `U` (User feature matrix)
  - `V` (Movie feature matrix)
- Learns hidden **latent features** representing user preferences and movie characteristics.
- Uses **gradient descent** to minimize prediction error.

**Key Steps:**
- Initialize `U` and `V` with small random values
- Update them iteratively based on rating errors
- Predict ratings as `R_pred = U × Vᵀ`

---

### 4️⃣ Content-Based Filtering
- Recommends movies **based on their genres**.
- Builds a **profile vector** for each user from the genres of movies they rated.
- Computes similarity between a user's profile and each movie's genre vector.

**Key Steps:**
- Extract **genre matrix** from movie data
- Build **user profiles** weighted by ratings
- Compute **cosine similarity** between profile and movies
- Recommend highest-similarity unseen movies

---

## ⚙️ Technologies Used
- **Python**  
- **Pandas** (data manipulation)  
- **NumPy** (matrix operations)  
- **scikit-learn** (cosine similarity)  

---

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn
