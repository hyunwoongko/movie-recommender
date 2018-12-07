def get_over_three_movie(rating_df):
    rating_df = rating_df.drop('userId', axis=1)
    rating_df = rating_df.drop('timestamp', axis=1)
    rating_df = rating_df.sort_values(by=['movieId'])
    rating_df = rating_df.groupby(['movieId'], as_index=False).mean()

    df_over_three = rating_df[rating_df['rating'] > 3]
    df_over_three = df_over_three.drop('rating', axis=1)
    return df_over_three


def get_movieId_tag_df(tag_df):
    tag_df = tag_df.drop('userId', axis=1)
    tag_df = tag_df.drop('timestamp', axis=1)
    return tag_df


def id_to_name(movieId, movie_df):
    return movie_df.loc[movie_df['movieId'] == movieId]


def id_to_names(movieId, movie_df):
    return movie_df.loc[movie_df['movieId'].isin(movieId)]
