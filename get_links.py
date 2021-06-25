""""
Collects all the youtube urls from the contestants.csv
returns all video ID's, or all the video ID's from a soecified year
"""

import pandas

filename = "../data/contestants.csv"
f = pandas.read_csv(filename)

# print(f)

def get_videoID(youtube_url):
    index = 0
    for i in range(len(youtube_url)):
        if youtube_url[i] == 'v' and youtube_url[i+1] == "=":
            index = i+2
            break
    return youtube_url[index:]


def get_list_with_ids():
    video_ids = []
    for i in range(len(f)):
        print(i)
        video_ids.append(get_videoID(f["youtube_url"][i]))
    # video_ids = video_ids.reverse()
    video_ids2 = video_ids[::-1]
    return video_ids2


def get_list_with_ids_per_year(year):
    video_ids = []
    for i in range(len(f)):
        if f["year"][i] == year:
            video_ids.append(get_videoID(f["youtube_url"][i]))
    # video_ids = video_ids.reverse()
    video_ids2 = video_ids[::-1]
    return video_ids2


def get_list_with_ids(year=None):
    video_ids = []
    for i in range(len(f)):
        if year: 
            if f["year"][i] == year:
                video_ids.append(get_videoID(f["youtube_url"][i]))
    # video_ids = video_ids.reverse()
    video_ids2 = video_ids[::-1]
    return video_ids2


def print_ids(video_ids):
    print("\n\n")
    for i in range(len(video_ids)):
        print("\n video: {}      id: {}".format(i, video_ids[i]))
    print("\n\n")



print(get_list_with_ids())


