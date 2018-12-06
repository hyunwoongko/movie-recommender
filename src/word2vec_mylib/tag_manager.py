from operator import itemgetter


def get_tag_by_user(tag_df, start_movie, start_user):
    result_array = []
    tmp_array = []

    for userId, movieId, tag, time in tag_df.values:

        global lower_tag
        if start_user == userId:
            if start_movie != movieId:
                sorted_list = sorted(tmp_array, key=itemgetter('time'))
                sorted_tag_list = []
                for i in sorted_list:
                    sorted_tag_list.append(i.get('tag'))
                result_array.append(list(sorted_tag_list))
                start_movie = movieId
                tmp_array.clear()

        else:
            sorted_list = sorted(tmp_array, key=itemgetter('time'))
            sorted_tag_list = []
            for i in sorted_list:
                sorted_tag_list.append(i.get('tag'))
            result_array.append(list(sorted_tag_list))
            start_user = userId
            start_movie = movieId
            tmp_array.clear()

        if not (type(tag) is float):
            lower_tag = tag.lower()
        data_dict = {'time': time, 'tag': lower_tag}
        tmp_array.append(data_dict)

    return result_array


def get_semantic_tag(test_movie_id, movieTag):
    tag_count = {}
    for movieId, tag in movieTag.values:
        global lower_tag
        if not (type(tag) is float):
            lower_tag = tag.lower()
        if test_movie_id == movieId:

            if not (lower_tag in tag_count):
                tag_count[lower_tag] = 1
            else:
                tempTagCount = tag_count[lower_tag]
                tag_count[lower_tag] = tempTagCount + 1

    sorted_list = sorted(tag_count.items(), key=lambda t: t[1], reverse=True)

    semantic = []
    for i in sorted_list:
        array = [i[0], i[1]]
        semantic.append(array)
        if len(semantic) > 2:
            break
    return semantic




