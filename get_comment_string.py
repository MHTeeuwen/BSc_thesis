# get_comment_string_from_csv()

import pandas as pd
import csv
import numpy as np

documents = pd.read_csv('../Data/combining/combined.csv', error_bad_lines=False)
documents.head()

documents2 = pd.read_csv('../Data/contestants.csv', error_bad_lines=False, skiprows=[i for i in range(1,1356)])
documents2.head()


filename = "../data/contestants.csv"
f = pd.read_csv(filename, skiprows=1356)

def get_videoID(youtube_url):
    index = 0
    for i in range(len(youtube_url)):
        if youtube_url[i] == 'v' and youtube_url[i+1] == "=":
            index = i+2
            break
    return youtube_url[index:]


def get_id_list(file):
    f = pd.read_csv(file)
    id_list = []
    for i in range(len(f)): 
        id_list.append(get_videoID(f["youtube_url"][i]))
    return id_list


def get_comment_string_from_csv(videoID):

    comment_string = ""
    for i in range(len(documents)):

        if documents["Video ID"][i] == videoID:
            comment_string += " " + str(documents["comment"][i])

    return comment_string


def get_placement(videoID):
    youtube_url = "https://youtube.com/watch?v=" + videoID
    # youtube_url = "https://youtube.com/watch?v=avBwSUYlTtI"
    place_final = ""

    for i in range(len(documents2)-1):
        # print(i, documents2["youtube_url"][i])
        if documents2["youtube_url"][i] == youtube_url:
            place_final = documents2["place_final"][i]
    
    # if place_final:
    #     if place_final != "NA":
    #         if int(place_final) < 6:
    #             return 1
    # else:
    #     return 0
    # if place_final < 6:
    #     return 1
    # else: 
    #     return 0
    
    if not isinstance(place_final, str):
        return place_final
    else: 
        return 0



# input: dataset, videoID and treshold. 
# output: 1 or 0
def is_top_contestant(videoID, treshold_top=0.75):
    place = get_placement(videoID)
    # print(place)
    if place >= treshold_top:
        return 1
    else: 
        return 0


list_with_all_ids = get_id_list("../data/contestants.csv")[1356:]

# # input: videoID, output: str met alle comments voor alleen deze video
# print(get_comment_string_from_csv("-7WpnSMEPjc"))

# # input: videoID, output: score in (0, 1). 0 is slecht. 1 is goed.
# print(get_placement("HV-eOhTS8Dw"))

# print(list_with_all_ids)




# # # #     create new DB

header = ['video_id', 'comment', 'is_top_contestant']
# list_with_some_ids = list_with_all_ids[-100:]
data = []

for i in list_with_all_ids:
# for i in list_with_some_ids:
    # print(i, get_comment_string_from_csv(i))
    new_data = []
    new_data.append(i)
    new_data.append(get_comment_string_from_csv(i))
    new_data.append(get_placement(i))
    data.append(new_data)
# print(data)





with open('NEWNEWNEW.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)







# doc = pd.read_csv('../Data/new_contestants.csv', error_bad_lines=False)
# doc.head()


# # open input CSV file as source
# # open output CSV file as result
# with open('../Data/new_contestants.csv', "r") as source:
#     reader = csv.reader(source)
      
#     with open("output.csv", "w") as result:
#         writer = csv.writer(result)
    
#         i = 0
#         for r in reader:
#             if i ==0:
#                 writer.writerow((r[0], r[1], "top_contestant", "videoID"))
#             i += 1
#             # Use CSV Index to remove a column from CSV
#             #r[3] = r['year']
#             if i > 1156:
#                 nl = r[19][28:]
#                 if r[8] != "NA":
#                     if int(r[8]) < 6:
#                         writer.writerow((r[0], r[1], 1, nl))
#                 else:
#                     writer.writerow((r[0], r[1], 0, nl))




# newdoc = pd.read_csv('output.csv', error_bad_lines=False)

                   
# # 40072608
# # for i in range(len(newdoc)):
# #     print(newdoc["videoID"][i])





# with open('output.csv', "r") as source:
#     reader = csv.reader(source)
      
#     with open("output2.csv", "w") as result:
#         writer = csv.writer(result)
#         i = 0
#         for r in reader:
#             if i == 0:
#                 writer.writerow((r[0], r[1], "top_contestant", "videoID", "comment"))
#             i += 1
#             # Use CSV Index to remove a column from CSV
#             #r[3] = r['year']
#             if i > 97:
#                 comment = get_comment_string_from_csv(r[3])
#                 writer.writerow((r[0], r[1], r[2], r[3], comment))

# get_comment_string_from_csv(videoID)