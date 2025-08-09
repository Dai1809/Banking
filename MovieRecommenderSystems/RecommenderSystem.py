import pandas as pd
import numpy as np

# Step 1: Load MovieLens 100K dataset
url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
columns = ['user_id', 'movie_id', 'rating', 'timestamp']
df = pd.read_csv(url, sep='\t', names=columns)
print(df.head())
# Step 2: Create user-item ratings matrix
ratings_matrix = df.pivot(index='user_id', columns='movie_id', values='rating')
ratings_matrix.fillna(0, inplace=True)
print(ratings_matrix.head())

# Load movie metadata
movie_titles = pd.read_csv(
    "https://files.grouplens.org/datasets/movielens/ml-100k/u.item",
    sep="|", encoding="latin-1", header=None,
    names=["movie_id", "title"], usecols=[0, 1]
)

movie_details = dict(zip(movie_titles['movie_id'],movie_titles['title']))

def cosinesim (u , v) :
    mask = (u != 0 ) & (v != 0 )
    if np.sum(mask)== 0:
        return 0
    u_common = u[mask]
    v_common = v[mask]

    num = np.dot(u_common,v_common)
    den = np.linalg.norm(u_common) * np.linalg.norm(v_common)

    return num / den

def Predict_Movie_Rating (userididx , movieid , ratings_matrix ,top_n) :
    userRatings = ratings_matrix.loc[userididx]
    if userRatings[movieid] != 0 :
        return userRatings[movieid] 
    

    similarities = []
    for otheruseridx in ratings_matrix.index :
        if otheruseridx == userididx :
            continue

        otheruserratings = ratings_matrix.loc[otheruseridx]
        if otheruserratings[movieid] == 0 :
            continue

        simi = cosinesim(userRatings , otheruserratings)

        similarities.append((otheruseridx , simi))

    similarities.sort(key=lambda x:x[1] , reverse= True)
    topusers = similarities[:top_n]
    numerator = 0.0
    denominator = 0.0

    for identifiedusers ,identifiedusersim  in  topusers:
        identifieduserrating = ratings_matrix.loc[identifiedusers,movieid]
        numerator += identifiedusersim * identifieduserrating
        denominator += abs(identifiedusersim)
         
    if denominator == 0 :
        return 0
    else :
        return numerator / denominator
        
finalmovieratingpred = Predict_Movie_Rating(1,1672,ratings_matrix,10)

print(f"Movie rating predicted to be {finalmovieratingpred:.2f}")

        
def recommendMovie (useridx , ratings_matrix , top_n_movies , top_n_users) :
    userratings = ratings_matrix.loc[useridx]

    unratedmovies = userratings[userratings == 0].index
    
    recommendedMovies = []
    for movieid in unratedmovies :
        predictedrating = Predict_Movie_Rating(useridx,movieid,ratings_matrix,top_n=top_n_users)
        recommendedMovies.append((movieid,predictedrating))
    
    recommendedMovies.sort(key=lambda x:x[1], reverse=True)

    return recommendedMovies[:top_n_movies]


recommendedmovieforauser = recommendMovie(5,ratings_matrix=ratings_matrix,top_n_movies=10,top_n_users=10)
for i , x in recommendedmovieforauser :
    title = movie_titles.get(i,'Unknown Movie')
    print(f"{title} is recommended for user with rating of {x:2f}")




    



