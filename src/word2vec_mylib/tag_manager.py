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


def get_all_tags_by_freq(test_movie_id, movieTag):
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

    all_tags_by_freq = []
    for i in sorted_list:
        array = [i[0], i[1]]
        all_tags_by_freq.append(array)
    return all_tags_by_freq


def get_TF_tags(all_tags):
    numberOfAll = 0
    temp = []

    for tag, numberOfTimes in all_tags:
        numberOfAll += numberOfTimes
        # 태그 가중치의 총합을 구함

    for tag, numberOfTimes in all_tags:
        weightForTag = (numberOfTimes / numberOfAll) * 10
        index = round(weightForTag)
        if index != 0:  # 그영화의 모든 태그중 가중치가 10퍼센트 이상인 태그라면
            temp.append(tag)  # 그 영화의 태그를 리스트에 담음

    return temp


def get_TF_tags_for_many(recommend_movies_with_tags):
    only_movieId = recommend_movies_with_tags.drop_duplicates('movieId')
    only_movieId = only_movieId.movieId.values

    recommend_TF = []  # 추천된 영화들의 TF 태그들 (가중치 10%이상)
    for movieId in only_movieId:
        recommend_movies_all_tags = get_all_tags_by_freq(movieId, recommend_movies_with_tags)  # 추천영화의 모든 태그를 구함
        recommend_movies_TF_tags = get_TF_tags(recommend_movies_all_tags)  # 추천영화의 TF 태그를 구함
        recommend_TF.append([movieId, recommend_movies_TF_tags])
    return recommend_TF


def get_recommend_tags(all_tags, model):
    numberOfAll = 0
    minTagWeight = 0
    temp = []

    for tag, numberOfTimes in all_tags:
        numberOfAll += numberOfTimes
        # 태그 가중치의 총합을 구함

    for tag, numberOfTimes in all_tags:
        weightForTag = (numberOfTimes / numberOfAll) * 10
        index = round(weightForTag)
        minTagWeight = index
        # 태그별 가중치를 구하고 모델을 사용할 최소가중치를 구함

    for tag, numberOfTimes in all_tags:
        weightForTag = (numberOfTimes / numberOfAll) * 10
        index = round(weightForTag)
        if index != 0:  # 그영화의 모든 태그중 가중치가 10퍼센트 이상인 태그라면
            temp.append(tag)  # 그 영화의 태그를 리스트에 담음

        if tag in model.wv.vocab and index > minTagWeight:
            for similar in model.wv.most_similar(tag):  # 아까 구한 모델을 사용할 최소가중치를 넘어서는 태그들은
                temp.append(similar[0])  # 모델에 넣어서 유사한 태그들도 가중치에 맞게 뽑아냄.
                index = index - 1  # 가중치만큼 반복해서 뽑음.
                if index == 0:
                    break

    distinct_list = list(set(temp))
    # 중복 제거
    additional_append = []
    if len(distinct_list) == 0:  # 태그리스트의 길이가 0이면 태그가 하나도 없는 영화임
        pass  # 그냥 지나침 ---> 어차피 추천을 해줄수 없음

    elif len(distinct_list) == 1:  # 태그 리스트의 길이가 1 (태그가 1개인 영화)
        for tag in distinct_list:  # 그 1개 태그와 유사한 7개 태그를 리스트에 담아냄
            if tag in model.wv.vocab:
                for new_item in model.wv.most_similar(tag):
                    additional_append.append(new_item[0])
                    if len(additional_append) >= 10:
                        break

    elif len(distinct_list) < 10:  # 태그가 1개이상10개 미만인 경우
        for tag in distinct_list:  # 태그별로 한개씩 담아서 10개 이상으로 만듬
            if tag in model.wv.vocab:  # 예) 태그 8개 -> 1번유사태그1개 , 2번유사태그1개 리스트에 담음
                new = list(model.wv.most_similar(tag)[0])
                additional_append.append(new[0])
            if len(additional_append) > 10:
                break

    for new_item in additional_append:  # 새로 꺼내온 아이템을 담아냄.
        distinct_list.append(new_item)
    distinct_list = list(set(distinct_list))

    if len(distinct_list) > 15:  # 만약 이러한 모든 과정을 거친뒤 태그의 갯수가 15개 이상이라면
        minTagWeight = minTagWeight + 1  # 모델을 사용할 태그가중치를 1올림
        distinct_list.clear()  # 다비우고
        temp.clear()  # 처음부터 다시

        for tag, numberOfTimes in all_tags:  # 위와 같은과정을 거치지만
            weightForTag = (numberOfTimes / numberOfAll) * 10  # 최소모델 사용 가중치를 1높혔기 때문에
            index = round(weightForTag)  # 아까보다 10퍼센트 더 높은 비중이 있는 태그들만
            if index != 0:  # 모델에 넣어서 비슷한 태그를 뽑고 아니면 그냥 그 태그만 담고, 모델을 사용하지 않음.
                temp.append(tag)

            if tag in model.wv.vocab and index > minTagWeight:
                for similar in model.wv.most_similar(tag):
                    temp.append(similar[0])
                    index = index - 1
                    if index == 0:
                        break
            distinct_list = list(set(temp))

    if len(distinct_list) > 15:  # 만약 이과정을 거쳤는데도 15개보다 많으면
        import random as rd  # 속도를 위해 랜덤으로 태그를 15개가 될때까지 뺌

        while len(distinct_list) > 15:  # 거의 모든 경우에 7개 이상 ~ 15개 미만의 태그 갯수가 보장됨
            toKill = rd.choice(distinct_list)  # 한가지 경우만 빼고 보장되는데
            distinct_list.remove(toKill)  # 태그가 1개나온경우, 그 태그가 min_count보다 적게 나왔을 경우는 추천을 해줄수 없음
            # 태그가 1개뿐인데 모델에 넣을수 없기 떄문에 이 경우는 어쩔 수 없음.
    return distinct_list
