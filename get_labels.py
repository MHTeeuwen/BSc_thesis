"""
insert a videoID. return normalized score based on contest placement (z score)

"""

""""
Collects all the youtube urls from the contestants.csv
returns all video ID's, or all the video ID's from a soecified year
"""

import pandas

filename = "../data/contestants.csv"
f = pandas.read_csv(filename)





# (place_final cell i)

# TODO: maak z score
# def get_z_score(data, index):
#     # data = np.array([6, 7, 7, 12, 13, 13, 15, 16, 19, 22])
#     return stats.zscore(data)[index]
# data = np.array([6, 7, 7, 12, 13, 13, 15, 16, 19, 22])
# OUTPUT -> [-1.394, -1.195, -1.195, -0.199, 0, 0, 0.398, 0.598, 1.195, 1.793]


def normalize_placement(place_final):
    return place_final / 40










def get_videoID(youtube_url):
    index = 0
    for i in range(len(youtube_url)):
        if youtube_url[i] == 'v' and youtube_url[i+1] == "=":
            index = i+2
            break
    return youtube_url[index:]


def get_list_with_ids_and_placement():
    video_ids = {}
    for i in reversed(range(len(f))):
        video_ids[get_videoID(f["youtube_url"][i])] = normalize_placement(f["place_final"][i])
        # video_ids.append((get_videoID(f["youtube_url"][i]), normalize_placement(f["place_final"][i])))
    return video_ids


# def get_list_with_ids_per_year(year):
#     video_ids = []
#     for i in range(len(f)):
#         if f["year"][i] == year:
#             video_ids.append(get_videoID(f["youtube_url"][i]))
#     # video_ids = video_ids.reverse()
#     video_ids2 = video_ids[::-1]
#     return video_ids2



# print(get_list_with_ids_and_placement())
# 6OjNzLaifFM 0.05

print(get_list_with_ids_and_placement()["6OjNzLaifFM"])
def get_placement_by_id(id):
    return get_list_with_ids_and_placement()[str(id)]

