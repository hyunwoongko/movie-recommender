import pandas as pd


def recommend(recommend_TF, testMovieTF_tags, model):
    recommend_movies_result = []
    zero_count_movies = []
    for movie in recommend_TF:
        similarity = 0
        count = 0

        for recommend_movies_tag in movie[1]:
            for current_movie_tag in testMovieTF_tags:
                count = count + 1
                if recommend_movies_tag in model.wv.vocab and current_movie_tag in model.wv.vocab:
                    similarity = similarity + model.wv.similarity(recommend_movies_tag, current_movie_tag)
        if count is not 0:
            recommend_movies_result.append([movie[0], similarity / (count / len(movie[1])), movie[1]])
        else:
            zero_count_movies.append([movie[0], count, movie[1]])

    for zero_movie in zero_count_movies:
        recommend_movies_result.append(zero_movie)

    recommend_movies_result.sort(key=lambda x: x[1], reverse=True)
    result_df = pd.DataFrame(recommend_movies_result, columns=['movieId', 'similarity', 'tags'])

    return result_df


def get_recommend_df(result_df, testMovieId):
    result_df = result_df[['movieId', 'similarity', 'rating', 'title', 'genres', 'tags']]
    result_df = result_df[result_df.movieId != testMovieId]  # 리스트에 현재 영화가 있었다면, 잘라냄

    factor = []
    for id, sim, rat, tit, gen, tag in result_df.values:
        factor.append(sim * rat)

    factor = pd.Series(factor)
    result_df['recommendFactor'] = factor.values
    result_df = result_df[['movieId', 'recommendFactor', 'title', 'genres', 'tags', 'similarity', 'rating']]
    result_df = result_df.sort_values(by=['recommendFactor'])
    result_df = result_df[::-1]
    return result_df
