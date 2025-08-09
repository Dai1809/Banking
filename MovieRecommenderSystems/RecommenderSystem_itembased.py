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
item_matrix = ratings_matrix.T



def cosinesim (u , v) :
    mask = (u != 0 ) & (v != 0 )
    if np.sum(mask)== 0:
        return 0
    u_common = u[mask]
    v_common = v[mask]

    num = np.dot(u_common,v_common)
    den = np.linalg.norm(u_common) * np.linalg.norm(v_common)

    return num / den

def Predict_Movie_Rating_using_itembased (userididx , movieid , ratings_matrix ,item_matrix ,top_n) :
    userRatings = ratings_matrix.loc[userididx]
    if userRatings[movieid] != 0 :
        return userRatings[movieid] 
    

    similarities = []

    rated_movies = userRatings[userRatings > 0].index

    for ratedmovieid in rated_movies :
        if ratedmovieid == movieid :
            continue

        simi = cosinesim(item_matrix.loc[ratedmovieid].values , item_matrix.loc[movieid].values)

        similarities.append((ratedmovieid , simi))

    similarities.sort(key=lambda x:x[1] , reverse= True)
    topmovies = similarities[:top_n]
    numerator = 0.0
    denominator = 0.0

    for identifiedmovies ,identifiedmoviesim  in  topmovies:
        identifieduserrating = ratings_matrix.loc[userididx,identifiedmovies]
        numerator += identifiedmoviesim * identifieduserrating
        denominator += abs(identifiedmoviesim)
         
    if denominator == 0 :
        return 0
    else :
        return numerator / denominator
        
finalmovieratingpred = Predict_Movie_Rating_using_itembased(1,1672,ratings_matrix,item_matrix,top_n=10)

print(f"Movie rating predicted to be {finalmovieratingpred:.2f}")

        
def recommendMovie (useridx , ratings_matrix ,item_matrix , top_n_movies , top_n_users) :
    userratings = ratings_matrix.loc[useridx]

    unratedmovies = userratings[userratings == 0].index
    
    recommendedMovies = []
    for movieid in unratedmovies :
        predictedrating = Predict_Movie_Rating_using_itembased(useridx,movieid,ratings_matrix,item_matrix,top_n=top_n_users)
        recommendedMovies.append((movieid,predictedrating))
    
    recommendedMovies.sort(key=lambda x:x[1], reverse=True)

    return recommendedMovies[:top_n_movies]


recommendedmovieforauser = recommendMovie(5,ratings_matrix=ratings_matrix,item_matrix=item_matrix,top_n_movies=10,top_n_users=10)
for i , x in recommendedmovieforauser :
    print(f"Movie ID {i} is recommended for user with rating of {x:2f}")




    



