import csv
import pandas as pd
import matplotlib.pyplot as plt

# with open('output13.csv', "r") as source:
#     reader = csv.reader(source)
      
#     with open("output14.csv", "w") as result:
#         writer = csv.writer(result)
#         i = 0
#         for r in reader:
#             if i == 0:
#                 # print("header")
#                 writer.writerow(r)
#             else: 
                
#                 else: 
#                     writer.writerow(r)
#             i += 1



import pandas as pd
#Read csv
df = pd.read_csv("output13.csv")

# Groupby and sum
df_new = df.groupby(["is_top_contestant"]).agg({"topic_1": "sum", "topic_2": "sum", "topic_3": "sum", "topic_4": "sum", "topic_5": "sum", "topic_6": "sum", "topic_7": "sum", "topic_8": "sum", "topic_9": "sum", "topic_10": "sum"}).sort_values(["is_top_contestant"]).reset_index()

# print(df_new)


# lijst = [
# lijst.append

x_1 = df_new["topic_1"]
x_2 = df_new["topic_2"]
x_3 = df_new["topic_3"]
x_4 = df_new["topic_4"]
x_5 = df_new["topic_5"]
x_6 = df_new["topic_6"]
x_7 = df_new["topic_7"]
x_8 = df_new["topic_8"]
x_9 = df_new["topic_9"]


y = [i for i in range(26)]
y = y[::-1]
# print(x_1)





# combined probabilities

# plt.figure()
# plt.scatter(y, x_1)
#         # plt.plot(fpr, tpr, color='darkorange', label=f"ROC curve (auc = {score})")
#         # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# plt.title(f"topic 1")
# plt.ylabel("Topic Probability")
# plt.xlabel("Placement in Contest")
# plt.legend(loc="lower right")
#         # plt.show()
#         # plt.savefig("../model_results/" + str(model) + '.png')
# plt.show()

# plt.scatter(y, x_2)
# plt.title(f"topic 2")
# plt.ylabel("Topic Probability")
# plt.xlabel("Placement in Contest")
# plt.legend(loc="lower right")
# plt.show()

# plt.scatter(y, x_3)
# plt.title(f"topic 3")
# plt.ylabel("Topic Probability")
# plt.xlabel("Placement in Contest")
# plt.legend(loc="lower right")
# plt.show()

# plt.scatter(y, x_4)
# plt.title(f"topic 4")
# plt.ylabel("Topic Probability")
# plt.xlabel("Placement in Contest")
# plt.legend(loc="lower right")
# plt.show()

# plt.scatter(y, x_5)
# plt.title(f"topic 5")
# plt.ylabel("Topic Probability")
# plt.xlabel("Placement in Contest")
# plt.legend(loc="lower right")
# plt.show()

# plt.scatter(y, x_6)
# plt.title(f"topic 6")
# plt.ylabel("Topic Probability")
# plt.xlabel("Placement in Contest")
# plt.legend(loc="lower right")
# plt.show()

# plt.scatter(y, x_7)
# plt.title(f"topic 7")
# plt.ylabel("Topic Probability")
# plt.xlabel("Placement in Contest")
# plt.legend(loc="lower right")
# plt.show()

# plt.scatter(y, x_8)
# plt.title(f"topic 8")
# plt.ylabel("Topic Probability")
# plt.xlabel("Placement in Contest")
# plt.legend(loc="lower right")
# plt.show()

# plt.scatter(y, x_9)
# plt.title(f"topic 9")
# plt.ylabel("Topic Probability")
# plt.xlabel("Placement in Contest")
# plt.legend(loc="lower right")
# plt.show()

# plt.scatter(y, x_10)
# plt.title(f"topic 10")
# plt.ylabel("Topic Probability")
# plt.xlabel("Placement in Contest")
# plt.legend(loc="lower right")
# plt.show()


# print(list(range(1,28)))
df2 = pd.read_csv("output13.csv")

# print(df2[range(1,28)])


# for i in range(len(df2)):
#     if df2["topic_1"][0] == 

# print(df2["topic_1"][24])



topic_1_1 = [0.1] + [df2["topic_1"][i] for i in range(25)]
print(topic_1_1)
topic_1_2 = [df2["topic_1"][i] for i in range(25,51)]
print(topic_1_2[0], topic_1_2[-1])
topic_1_3 = [df2["topic_1"][i] for i in range(51,77)]
print(topic_1_3[0], topic_1_3[-1])
topic_1_4 = [df2["topic_1"][i] for i in range(77,103)]
# print(topic_1_4[0], topic_1_4[-1])
topic_1_5 = [df2["topic_1"][i] for i in range(104,130)]
# print(topic_1_5[0], topic_1_5[-1])

topic_2_1 = [0.1] + [df2["topic_2"][i] for i in range(25)]
topic_2_2 = [df2["topic_2"][i] for i in range(25,51)]
topic_2_3 = [df2["topic_2"][i] for i in range(51,77)]
topic_2_4 = [df2["topic_2"][i] for i in range(77,103)]
topic_2_5 = [df2["topic_2"][i] for i in range(104,130)]

topic_3_1 = [0.1] + [df2["topic_3"][i] for i in range(25)]
topic_3_2 = [df2["topic_3"][i] for i in range(25,51)]
topic_3_3 = [df2["topic_3"][i] for i in range(51,77)]
topic_3_4 = [df2["topic_3"][i] for i in range(77,103)]
topic_3_5 = [df2["topic_3"][i] for i in range(104,130)]

topic_4_1 = [0.1] + [df2["topic_4"][i] for i in range(25)]
topic_4_2 = [df2["topic_4"][i] for i in range(25,51)]
topic_4_3 = [df2["topic_4"][i] for i in range(51,77)]
topic_4_4 = [df2["topic_4"][i] for i in range(77,103)]
topic_4_5 = [df2["topic_4"][i] for i in range(104,130)]

topic_5_1 = [0.1] + [df2["topic_5"][i] for i in range(25)]
topic_5_2 = [df2["topic_5"][i] for i in range(25,51)]
topic_5_3 = [df2["topic_5"][i] for i in range(51,77)]
topic_5_4 = [df2["topic_5"][i] for i in range(77,103)]
topic_5_5 = [df2["topic_5"][i] for i in range(104,130)]

topic_6_1 = [0.1] + [df2["topic_6"][i] for i in range(25)]
topic_6_2 = [df2["topic_6"][i] for i in range(25,51)]
topic_6_3 = [df2["topic_6"][i] for i in range(51,77)]
topic_6_4 = [df2["topic_6"][i] for i in range(77,103)]
topic_6_5 = [df2["topic_6"][i] for i in range(104,130)]

topic_7_1 = [0.1] + [df2["topic_7"][i] for i in range(25)]
topic_7_2 = [df2["topic_7"][i] for i in range(25,51)]
topic_7_3 = [df2["topic_7"][i] for i in range(51,77)]
topic_7_4 = [df2["topic_7"][i] for i in range(77,103)]
topic_7_5 = [df2["topic_7"][i] for i in range(104,130)]

topic_8_1 = [0.1] + [df2["topic_8"][i] for i in range(25)]
topic_8_2 = [df2["topic_8"][i] for i in range(25,51)]
topic_8_3 = [df2["topic_8"][i] for i in range(51,77)]
topic_8_4 = [df2["topic_8"][i] for i in range(77,103)]
topic_8_5 = [df2["topic_8"][i] for i in range(104,130)]

topic_9_1 = [0.1] + [df2["topic_9"][i] for i in range(25)]
topic_9_2 = [df2["topic_9"][i] for i in range(25,51)]
topic_9_3 = [df2["topic_9"][i] for i in range(51,77)]
topic_9_4 = [df2["topic_9"][i] for i in range(77,103)]
topic_9_5 = [df2["topic_9"][i] for i in range(104,130)]

topic_10_1 = [0.1] + [df2["topic_10"][i] for i in range(25)]
topic_10_2 = [df2["topic_10"][i] for i in range(25,51)]
topic_10_3 = [df2["topic_10"][i] for i in range(51,77)]
topic_10_4 = [df2["topic_10"][i] for i in range(77,103)]
topic_10_5 = [df2["topic_10"][i] for i in range(104,130)]





plt.figure()
# plt.xlim(, 5)
plt.ylim(0, 0.75)
plt.scatter(y, topic_1_1, c='b', marker='.', label='2015')
plt.scatter(y, topic_1_2, c='r', marker='.', label='2016')
plt.scatter(y, topic_1_3, c='g', marker='.', label='2017')
plt.scatter(y, topic_1_4, c='y', marker='.', label='2018')
plt.scatter(y, topic_1_5, c='c', marker='.', label='2019')
plt.title(f"Topic 1")
plt.ylabel("Topic Probability")
plt.xlabel("Placement in Contest")
plt.legend(loc="upper right")
plt.savefig("../plots/topic_1.png")
plt.show()

plt.figure()
plt.ylim(0, 0.75)
plt.scatter(y, topic_2_1, c='b', marker='.', label='2015')
plt.scatter(y, topic_2_2, c='r', marker='.', label='2016')
plt.scatter(y, topic_2_3, c='g', marker='.', label='2017')
plt.scatter(y, topic_2_4, c='y', marker='.', label='2018')
plt.scatter(y, topic_2_5, c='c', marker='.', label='2019')
plt.title(f"Topic 2")
plt.ylabel("Topic Probability")
plt.xlabel("Placement in Contest")
plt.legend(loc="upper right")
plt.savefig("../plots/topic_2.png")
plt.show()

plt.figure()
plt.ylim(0, 0.75)
plt.scatter(y, topic_3_1, c='b', marker='.', label='2015')
plt.scatter(y, topic_3_2, c='r', marker='.', label='2016')
plt.scatter(y, topic_3_3, c='g', marker='.', label='2017')
plt.scatter(y, topic_3_4, c='y', marker='.', label='2018')
plt.scatter(y, topic_3_5, c='c', marker='.', label='2019')
plt.title(f"Topic 3")
plt.ylabel("Topic Probability")
plt.xlabel("Placement in Contest")
plt.legend(loc="upper right")
plt.savefig("../plots/topic_3.png")
plt.show()

plt.figure()
plt.ylim(0, 0.75)
plt.scatter(y, topic_4_1, c='b', marker='.', label='2015')
plt.scatter(y, topic_4_2, c='r', marker='.', label='2016')
plt.scatter(y, topic_4_3, c='g', marker='.', label='2017')
plt.scatter(y, topic_4_4, c='y', marker='.', label='2018')
plt.scatter(y, topic_4_5, c='c', marker='.', label='2019')
plt.title(f"Topic 4")
plt.ylabel("Topic Probability")
plt.xlabel("Placement in Contest")
plt.legend(loc="upper right")
plt.savefig("../plots/topic_4.png")
plt.show()

plt.figure()
plt.ylim(0, 0.75)
plt.scatter(y, topic_5_1, c='b', marker='.', label='2015')
plt.scatter(y, topic_5_2, c='r', marker='.', label='2016')
plt.scatter(y, topic_5_3, c='g', marker='.', label='2017')
plt.scatter(y, topic_5_4, c='y', marker='.', label='2018')
plt.scatter(y, topic_5_5, c='c', marker='.', label='2019')
plt.title(f"Topic 5")
plt.ylabel("Topic Probability")
plt.xlabel("Placement in Contest")
plt.legend(loc="upper right")
plt.savefig("../plots/topic_5.png")
plt.show()

plt.figure()
plt.ylim(0, 0.75)
plt.scatter(y, topic_6_1, c='b', marker='.', label='2015')
plt.scatter(y, topic_6_2, c='r', marker='.', label='2016')
plt.scatter(y, topic_6_3, c='g', marker='.', label='2017')
plt.scatter(y, topic_6_4, c='y', marker='.', label='2018')
plt.scatter(y, topic_6_5, c='c', marker='.', label='2019')
plt.title(f"Topic 6")
plt.ylabel("Topic Probability")
plt.xlabel("Placement in Contest")
plt.legend(loc="upper right")
plt.savefig("../plots/topic_6.png")
plt.show()

plt.figure()
plt.ylim(0, 0.75)
plt.scatter(y, topic_7_1, c='b', marker='.', label='2015')
plt.scatter(y, topic_7_2, c='r', marker='.', label='2016')
plt.scatter(y, topic_7_3, c='g', marker='.', label='2017')
plt.scatter(y, topic_7_4, c='y', marker='.', label='2018')
plt.scatter(y, topic_7_5, c='c', marker='.', label='2019')
plt.title(f"Topic 7")
plt.ylabel("Topic Probability")
plt.xlabel("Placement in Contest")
plt.legend(loc="upper right")
plt.savefig("../plots/topic_7.png")
plt.show()

plt.figure()
plt.ylim(0, 0.75)
plt.scatter(y, topic_8_1, c='b', marker='.', label='2015')
plt.scatter(y, topic_8_2, c='r', marker='.', label='2016')
plt.scatter(y, topic_8_3, c='g', marker='.', label='2017')
plt.scatter(y, topic_8_4, c='y', marker='.', label='2018')
plt.scatter(y, topic_8_5, c='c', marker='.', label='2019')
plt.title(f"Topic 8")
plt.ylabel("Topic Probability")
plt.xlabel("Placement in Contest")
plt.legend(loc="upper right")
plt.savefig("../plots/topic_8.png")
plt.show()


plt.figure()
plt.ylim(0, 0.75)
plt.scatter(y, topic_9_1, c='b', marker='.', label='2015')
plt.scatter(y, topic_9_2, c='r', marker='.', label='2016')
plt.scatter(y, topic_9_3, c='g', marker='.', label='2017')
plt.scatter(y, topic_9_4, c='y', marker='.', label='2018')
plt.scatter(y, topic_9_5, c='c', marker='.', label='2019')
plt.title(f"Topic 9")
plt.ylabel("Topic Probability")
plt.xlabel("Placement in Contest")
plt.legend(loc="upper right")
plt.savefig("../plots/topic_9.png")
plt.show()

plt.figure()
plt.ylim(0, 0.75)
plt.scatter(y, topic_10_1, c='b', marker='.', label='2015')
plt.scatter(y, topic_10_2, c='r', marker='.', label='2016')
plt.scatter(y, topic_10_3, c='g', marker='.', label='2017')
plt.scatter(y, topic_10_4, c='y', marker='.', label='2018')
plt.scatter(y, topic_10_5, c='c', marker='.', label='2019')
plt.title(f"Topic 10")
plt.ylabel("Topic Probability")
plt.xlabel("Placement in Contest")
plt.legend(loc="upper right")
plt.savefig("../plots/topic_10.png")
plt.show()


plt.figure()
