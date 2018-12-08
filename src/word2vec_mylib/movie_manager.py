import pandas as pd


def get_higher_rating_movie(rating_df):
    rating_df = rating_df.drop('userId', axis=1)
    rating_df = rating_df.drop('timestamp', axis=1)
    rating_df = rating_df.sort_values(by=['movieId'])
    rating_df = rating_df.groupby(['movieId'], as_index=False).mean()
    df_over_three = rating_df[rating_df['rating'] > 3]
    return df_over_three


def cut_userId_timestamp(tag_df):
    tag_df = tag_df.drop('userId', axis=1)
    tag_df = tag_df.drop('timestamp', axis=1)
    return tag_df


def get_recommend_movie_list(tag_df, higher_rating, testMovie_model_tags):
    recommend_movies = tag_df.drop_duplicates('movieId')  # 무비아이디를 축으로 중복 영화 자름
    recommend_movies = pd.merge(recommend_movies, higher_rating)  # 3점 이상인 영화 리스트와 join
    recommend_movies = recommend_movies[recommend_movies['tag'].isin(testMovie_model_tags)]  # 모델태그리스트에 있는거 하나라도 있는영화만
    recommend_movies = recommend_movies.drop('tag', axis=1)  # 이렇게 추출하면 유사태그들을 제외한 다른태그들이 잘림
    recommend_movies_array = recommend_movies.movieId.values  # 여기에서는 무비 아이디만 획득하는 것으로 만족함
    return recommend_movies_array


def get_recommend_movie_with_all_tags(tag_df, recommend_movies_array):
    recommend_movies_with_tags = tag_df[tag_df['movieId'].isin(recommend_movies_array)]  # 무비아이디로 모든 태그를 긁어옴
    recommend_movies_with_tags = recommend_movies_with_tags.sort_values(by=['movieId'])  # 무비아이디 순으로 정렬
    return recommend_movies_with_tags


def id_to_name(movieId, movie_df):
    return movie_df.loc[movie_df['movieId'] == movieId]


def id_to_names(movieId, movie_df):
    return movie_df.loc[movie_df['movieId'].isin(movieId)]
