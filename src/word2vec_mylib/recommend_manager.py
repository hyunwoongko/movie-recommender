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
