import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


rating_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
rating_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
rating_df = pd.read_csv(rating_url, sep='\t', names=rating_columns)
rating_df.drop('timestamp', axis=1, inplace=True)
print(rating_df.head())

movie_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
movie_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + \
             [f'genre_{i}' for i in range(19)]  # 19 genres
movies_df = pd.read_csv(movie_url, sep='|', names=movie_cols, encoding='latin-1')
genre_cols = [f'genre_{i}' for i in range(19)]
movies_df = movies_df[['movie_id', 'title'] + genre_cols ]


merged_df = pd.merge(rating_df, movies_df, on='movie_id')



user_profiles={}

for userid in merged_df['user_id'].unique():

    user_data = merged_df[merged_df['user_id']==userid]

    genre_matrix = user_data[genre_cols].values
    ratings = user_data['rating'].values.reshape(-1,1)

    user_profile = np.sum(genre_matrix * ratings,axis=0)

    if user_profile.sum() != 0 :
        user_profile = user_profile / user_profile.sum()

    user_profiles[userid] = user_profile


user_profile_df = pd.DataFrame.from_dict(user_profiles,orient='index',columns=genre_cols)


movie_genre_matrix = movies_df[genre_cols].values
user_sim_score={}
for user_id in user_profile_df.index:
    user_vector = user_profile_df.loc[user_id].values.reshape(1,-1)
    user_sim = cosine_similarity(user_vector,movie_genre_matrix)[0]
    user_sim_score[user_id]=user_sim

top_n = 5
final_recommendations={}

for userid in user_profile_df.index:
        
    watched_movies =rating_df[rating_df['user_id']==user_id]['movie_id'].values
    
    sim_scores = user_sim_score[user_id]
    
    movie_index_sorted = np.argsort(sim_scores)[::-1]
    
    movie_index_sorted = [idx for idx in movie_index_sorted if movies_df.iloc[idx]['movie_id'] not in watched_movies]

    final_recommendations[user_id]= movies_df.iloc[movie_index_sorted[:top_n]]['title'].tolist()


