def get_over_three_movie(rating_df):
    rating_df = rating_df.drop('userId', axis=1)
    rating_df = rating_df.drop('timestamp', axis=1)
    rating_df = rating_df.sort_values(by=['movieId'])
    rating_df = rating_df.groupby(['movieId'], as_index=False).mean()

    df_over_three = rating_df[rating_df['rating'] > 3]
    df_over_three = df_over_three.drop('rating', axis=1)
    return df_over_three


def recommend(semantic_df, get_movie_tag, movieId):
    result = []
    tags = []

    for i in range(0, len(get_movie_tag(movieId))):
        tag = get_movie_tag(movieId)[i][0]
        tags.append(tag)

    recommend_df = semantic_df[semantic_df['tag'].isin(tags)]
    recommend_df = recommend_df.drop('tag', axis=1)
    recommend_df = recommend_df.drop_duplicates()
    for data in recommend_df.values:
        result.append(data[0])

    if movieId in result:
        result.remove(movieId)
    return result


def id_to_name(movieId, movie_df):
    return movie_df.loc[movie_df['movieId'] == movieId]


def id_to_names(movieId, movie_df):
    return movie_df.loc[movie_df['movieId'].isin(movieId)]
