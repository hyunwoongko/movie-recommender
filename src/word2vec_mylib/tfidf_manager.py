def get_top3_list(tfidfArray):
    global top3list
    for n in range(len(tfidfArray)):
        top3list = sorted(range(len(tfidfArray[n])), key=lambda i: tfidfArray[n][i])[-3:]
    return top3list




def get_tfidf_sentence(movie_tag_df, tag_sentence):
    strTemp = ""
    tempMovieId = 1
    for movieId, tag in movie_tag_df.values:
        if movieId != tempMovieId:
            tag_sentence.append(strTemp)
            strTemp = str(movieId)
            tempMovieId = movieId
        if not (type(tag) is float):
            strTemp = strTemp + " " + tag.lower()
        else:
            strTemp = strTemp + " " + str(tag)

    return tag_sentence
