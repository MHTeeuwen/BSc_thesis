# - Maak scatterplot voor elke top feature (topic) in the final 
# Y final place 1/40
# X topic prob
# 10 plots, one for every topic.
# Different colours per year

import pandas as pd
import csv


filename = "../data/contestants.csv"
f = pd.read_csv(filename, skiprows=[i for i in range(1,1356)])


def find_placement_by_ID(videoID):
    final_place = ""
    vid = "https://youtube.com/watch?v=" + videoID
    # final_place = 6

    for i in range(len(f)):
        # if f["place_final"][i] > 0 and f["place_final"][i] < 30:
        # print(i)
        if f["youtube_url"][i] == vid: 
            # print(f["youtube_url"][i])
            # print(f["place_final"][i])
            final_place = f["place_final"][i]

    return final_place
    # print(f["place_final"][0])

# print(find_placement_by_ID("QLrXmTB8OaY"))
# print(type(find_placement_by_ID("ya1r_nFHiCQ")))




with open('output12.csv', "r") as source:
    reader = csv.reader(source)
      
    with open("output13.csv", "w") as result:
        writer = csv.writer(result)
        i = 0
        for r in reader:
            if i == 0:
                writer.writerow(r)
            else:
                if int(r[1]) < 6:
                    r.append(1)
                elif int(r[1]) >= 6:
                    r.append(0)
                writer.writerow(r)
                # # print(type(r[1]))
                # if len(r[1]) == 3:
                #     r[1] = r[1][0] 
                # elif len(r[1]) == 4:
                #     r[1] = r[1][:2]
                # else: 
                #     r[1] = 27
                # try:
                #     z = int(r[1])
                #     if z < 27:
                #         writer.writerow(r)
                #     else:
                #         r[1] = 27
                #         # r.append(find_placement_by_ID(r[0]))
                #         writer.writerow(r)
                # except:
                #     continue
            i += 1
