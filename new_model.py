import pandas as pd
import gensim
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import csv
print("\n\n\n\n\n\n\n\n\n")


documents = pd.read_csv('../Data/combining/combined.csv', error_bad_lines=False)
documents.head()
single_doc = pd.read_csv('../Data/2017_comments/2017_comments_ZSHc7iDuBCQ.csv', error_bad_lines=False)
documents.head()

documents2 = pd.read_csv('../Data/contestants.csv', error_bad_lines=False)
documents2.head()


num_topics = 15
# Use CountVectorizor to find three letter tokens, remove stop_words, 
# remove tokens that don't appear in at least 20 documents,
# remove tokens that appear in more than 20% of the documents
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', 
                       token_pattern='(?u)\\b\\w\\w\\w+\\b')
# Fit and transform
X = vect.fit_transform(documents.comment.values.astype('U'))
# Convert sparse matrix to gensim corpus.
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False) 
# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())
# Use the gensim.models.ldamodel.LdaModel constructor to estimate 
# LDA model parameters on the corpus, and save to the variable `ldamodel`
ldamodel = gensim.models.LdaMulticore(corpus=corpus, id2word=id_map, passes=2,
                                               random_state=5, num_topics=num_topics, workers=2)

for idx, topic in ldamodel.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic))
    print("\n")


def get_placement(videoID):
    youtube_url = "https://youtube.com/watch?v=" + videoID
    # youtube_url = "https://youtube.com/watch?v=avBwSUYlTtI"
    place_final = ""

    for i in range(len(documents2)-1):
        # print(i, documents2["youtube_url"][i])
        if documents2["youtube_url"][i] == youtube_url:
            place_final = documents2["place_final"][i]

    return np.int_(place_final)



def get_comment_string_from_csv(videoID):
    # videoID = "-7WpnSMEPjc"
    comment_string = ""
    for i in range(len(documents)):
        if documents["Video ID"][i] == videoID:
            comment_string += " " + str(documents["comment"][i])
    return comment_string


def get_topic_distribution(string_input):
    string_input = [string_input]
    # Fit and transform
    X = vect.transform(string_input)
    # Convert sparse matrix to gensim corpus.
    corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
    output = list(ldamodel[corpus])[0]
    newlist = []
    for i in output:
        newlist.append(i[1])
    return newlist


def get_topic_prediction(my_document):
    string_input = [my_document]
    X = vect.transform(string_input)
    # Convert sparse matrix to gensim corpus.
    corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
    output = list(ldamodel[corpus])[0]
    topics = sorted(output,key=lambda x:x[1],reverse=True)
    return topics[0][0]


# def get_feature_vec(output_topic_distribution):
#     newlist = []
#     for i in output_topic_distribution:
#         newlist.append(i[1])
#     return new_list

#     feature_vec = []
#     for i in output_topic_distribution:
#         feature_vec.append(float(i[1]))
#     return feature_vec




# videoID = "_9XsCqGg8ls"
# my_document = get_comment_string_from_csv(videoID)
# topic_distribution = get_topic_distribution(my_document)
# topic_prediction = get_topic_prediction(my_document)

# print(len(topic_distribution))






# with open('NEWNEWNEW.csv', "r") as source:
#     reader = csv.reader(source)
      
#     with open("output9.csv", "w") as result:
#         writer = csv.writer(result)
#         i = 0
#         for r in reader:
#             print(get_topic_distribution(r[1]))
#             # print(type(r[1]))
#             # print(get_feature_vec(r[0]))
#             if i == 0:
#                 # print(r[0])
#                 writer.writerow((r[0], r[1], r[2], "topic_1", "topic_2", "topic_3", "topic_4", "topic_5", "topic_6", "topic_7", "topic_8", "topic_9", "topic_10" ))
#             else: 
#                 x = float(0.10000)
#                 feature_vec = get_topic_distribution(r[1])
#                 if len(feature_vec) == 10:
#                     writer.writerow((r[0], r[1], r[2], float(feature_vec[0]), float(feature_vec[1]), float(feature_vec[2]), float(feature_vec[3]), float(feature_vec[4]), float(feature_vec[5]), float(feature_vec[6]),  float(feature_vec[7]), float(feature_vec[8]), float(feature_vec[9])))
#                 elif len(feature_vec) == 9:
#                     writer.writerow((r[0], r[1], r[2], float(feature_vec[0]), float(feature_vec[1]), float(feature_vec[2]), float(feature_vec[3]), float(feature_vec[4]), float(feature_vec[5]), float(feature_vec[6]),  float(feature_vec[7]), float(feature_vec[8]), x ))
#                 elif len(feature_vec) == 8: 
#                     writer.writerow((r[0], r[1], r[2], float(feature_vec[0]), float(feature_vec[1]), float(feature_vec[2]), float(feature_vec[3]), float(feature_vec[4]), float(feature_vec[5]), float(feature_vec[6]),  float(feature_vec[7]), x, x ))
#                 elif len(feature_vec) == 7: 
#                     writer.writerow((r[0], r[1], r[2], float(feature_vec[0]), float(feature_vec[1]), float(feature_vec[2]), float(feature_vec[3]), float(feature_vec[4]), float(feature_vec[5]), float(feature_vec[6]), x, x, x ))
#                 else:
#                     writer.writerow((r[0], r[1], r[2], x, x, x, x, x, x, x, x, x, x))
#             i += 1



# with open('output14.csv', "r") as source:
#     reader = csv.reader(source)
#     if num_topics == 5:
#         with open("5topics.csv", "w") as result:
#             writer = csv.writer(result)
#             i = 0
#             for r in reader:
#                 videoID = r[0]
#                 my_document = get_comment_string_from_csv(videoID)
#                 td = get_topic_distribution(my_document)
#                 if i == 0:
#                     writer.writerow((r[0], r[1], "topic_1", "topic_2", "topic_3", "topic_4", "topic_5", "target"  ))
#                 else: 
#                     if len(td) == 5:
#                         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], r[-1]))
#                     elif len(td) == 4:
#                         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], 0.2, r[-1]))
#                 i += 1



# with open('output14.csv', "r") as source:
#     reader = csv.reader(source)
#     if num_topics == 15:
#         with open("15topics.csv", "w") as result:
#             writer = csv.writer(result)
#             i = 0
            
#             for r in reader:
#                 videoID = r[0]
#                 my_document = get_comment_string_from_csv(videoID)
#                 td = get_topic_distribution(my_document)
#                 if i == 0:
                    
#                     writer.writerow((r[0], r[1], "topic_1", "topic_2", "topic_3", "topic_4", "topic_5", "topic_6", "topic_7", "topic_8", "topic_9", "topic_10", "topic_11", "topic_12",  "topic_13",  "topic_14",  "topic_15", "target" ))
#                     # writer.writerow((r[0], r[1], "topic_1", "topic_2", "topic_3", "topic_4", "topic_5", "topic_6", "topic_7", "topic_8", "topic_9", "topic_10", "topic_11", "topic_12",  "topic_13",  "topic_14",  "topic_15"    ))
#                 else: 
#                     if len(td) == 15:
#                         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], td[9], td[10], td[11], td[12], td[13], td[14], r[-1]))
#                     elif len(td) == 14: 
#                         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], td[9], td[10], td[11], td[12], td[13], 0.0667, r[-1]))
#                     elif len(td) == 13: 
#                         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], td[9], td[10], td[11], td[12], 0.0667, 0.0667, r[-1]))
#                     elif len(td) == 12:
#                         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], td[9], td[10], td[11], 0.0667, 0.0667, 0.0667, r[-1]))
#                     elif len(td) == 11:  
#                         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], td[9], td[10], 0.0667, 0.0667, 0.0667, 0.0667, r[-1]))
#                     elif len(td) == 10:
#                         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], td[9], 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, r[-1]))
#                     elif len(td) == 9:
#                         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, r[-1]))
#                     elif len(td) == 8:
#                         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], 0.0667, 0.0667, 0.0667, 0.1, 0.0667, 0.0667, 0.0667, r[-1]))
#                     elif len(td) == 7:    
#                         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, r[-1]))
#                     else:
#                         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, r[-1]))
#                 i += 1




# with open('output14.csv', "r") as source:
#     reader = csv.reader(source)
#     if num_topics == 20: 
#         with open("20topics.csv", "w") as result:
#             writer = csv.writer(result)
#             i = 0
            
#             for r in reader:
#                 videoID = r[0]
#                 my_document = get_comment_string_from_csv(videoID)
#                 td = get_topic_distribution(my_document)
#                 if i == 0:
                    
#                     writer.writerow((r[0], r[1], "topic_1", "topic_2", "topic_3", "topic_4", "topic_5", "topic_6", "topic_7", "topic_8", "topic_9", "topic_10", "topic_11", "topic_12",  "topic_13",  "topic_14",  "topic_15",  "topic_16",  "topic_17",  "topic_18","topic_19", "target"     ))
#                     # writer.writerow((r[0], r[1], "topic_1", "topic_2", "topic_3", "topic_4", "topic_5", "topic_6", "topic_7", "topic_8", "topic_9", "topic_10", "topic_11", "topic_12",  "topic_13",  "topic_14",  "topic_15"    ))
#                 else: 
#                     if len(td) == 19:
#                         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], td[9], td[10], td[11], td[12], td[13], td[14], td[15], td[16], td[17], td[18], r[-1]))
#                     elif len(td) == 18: 
#                         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], td[9], td[10], td[11], td[12], td[13], td[14], td[15], td[16], td[17], 0.05, r[-1]))
#                     elif len(td) == 17: 
#                         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], td[9], td[10], td[11], td[12], td[13], td[14], td[15], td[16], 0.05, 0.05, r[-1]))
#                     elif len(td) == 16:
#                         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], td[9], td[10], td[11], td[12], td[13], td[14], td[15], 0.05, 0.05, 0.05, r[-1]))
                    # elif len(td) == 15:
                #         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], td[9], td[10], td[11], td[12], td[13], td[14], 0.05, 0.05, 0.05, 0.05, r[-1]))
                #     elif len(td) == 14: 
                #         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], td[9], td[10], td[11], td[12], td[13], 0.05, 0.05, 0.05, 0.05, 0.05, r[-1]))
                #     elif len(td) == 13: 
                #         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], td[9], td[10], td[11], td[12], 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, r[-1]))
                #     elif len(td) == 12:
                #         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], td[9], td[10], td[11], 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, r[-1]))
                #     elif len(td) == 11:  
                #         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], td[9], td[10], 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, r[-1]))
                #     elif len(td) == 10:
                #         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], td[9], 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, r[-1]))
                #     elif len(td) == 9:
                #         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], td[8], 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, r[-1]))
                #     elif len(td) == 8:
                #         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, r[-1]))
                #     elif len(td) == 7:    
                #         writer.writerow((r[0], r[1], td[0], td[1], td[2], td[3], td[4], td[5], td[6], 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, r[-1]))
                # i += 1



# header = ["VideoID", "topic1", "topic2", "topic3", "topic4", "topic5", "topic6", "topic7", "topic8", "topic9", "topic10", "topic11", "target"]
# data = [
#     ["-7WpnSMEPjc", 0.0779608, 0.10468809, 0.10574545, 0.037954066, 0.15815903, 0.10431655, 0.17747784, 0.07634576, 0.026331034, 0.13102141, 2.305843009213694e+17, 1], 
#     ["Qotooj7ODCM", 0.13592245, 0.25728467, 0.06789132, 0.10358463, 0.0001143, 0.020208843, 0.06392356, 0.1185572, 0.1161308, 0.11614306, 0.975, 1],
#     ["_9XsCqGg8ls", 0.08992998, 0.047209337, 0.1862531, 0.032720137, 0.13644679, 0.10389945, 0.03859395, 0.097721264, 0.13913667, 0.12808937, 0.725, 0],
#     ["ffwSMEPjc", 0.0779608, 0.10468809, 0.10574545, 0.037954066, 0.15815903, 0.10431655, 0.17747784, 0.07634576, 0.026331034, 0.13102141, 2.305843009213694e+17, 0], 
#     ["Q3tooj7ODCM", 0.13592245, 0.25728467, 0.06789132, 0.10358463, 0.0001143, 0.020208843, 0.06392356, 0.1185572, 0.1161308, 0.11614306, 0.975, 0],
#     ["49XsCqGg8ls", 0.08992998, 0.047209337, 0.1862531, 0.032720137, 0.13644679, 0.10389945, 0.03859395, 0.097721264, 0.13913667, 0.12808937, 0.725, 0],
#     ["h7WpnSMEPjc", 0.0779608, 0.10468809, 0.10574545, 0.037954066, 0.15815903, 0.10431655, 0.17747784, 0.07634576, 0.026331034, 0.13102141, 2.305843009213694e+17, 1], 
#     ["Qoto77j7ODCM", 0.13592245, 0.25728467, 0.06789132, 0.10358463, 0.0001143, 0.020208843, 0.06392356, 0.1185572, 0.1161308, 0.11614306, 0.975, 1],
#     ["_9sCqGg8ls", 0.08992998, 0.047209337, 0.1862531, 0.032720137, 0.13644679, 0.10389945, 0.03859395, 0.097721264, 0.13913667, 0.12808937, 0.725, 1]
# ]

# with open('test.csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)

#     # write the header
#     writer.writerow(header)

#     # write multiple rows
#     writer.writerows(data)


# ndata = pd.read_csv('output9.csv')
  
# # display 
# # print("Original 'input.csv' CSV Data: \n")
# # print(data)
  
# # drop function which is used in removing or deleting rows or columns from the CSV files
# ndata.drop('comment', inplace=True, axis=1)
  
# # display 
# print("\nCSV Data after deleting the column 'year':\n")
# print(ndata)






# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# # data = pd.read_csv('workfile.csv', error_bad_lines=False, index_col='videoID')
# # data.head()

# if num_topics == 10:
#     filename = 'output14.csv'
# else:
#     filename = str(num_topics) + "topics.csv"



# df = pd.read_csv(filename, index_col='video_id')
# df.drop('is_top_contestant', inplace=True, axis=1)

# trues = df.loc[df['target'] == 1]
# falses = df.loc[df['target'] != 1].sample(frac=1)[:len(trues)]
# data = pd.concat([trues, falses], ignore_index=True).sample(frac=1)
# data.head()

# y = data['target']
# X = data.drop('target', axis=1)


# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)


# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)

# # print("\n\n X_train: {} \n\n\n X_valid: {} \n\n\n y_train: {} y_valid: {}".format(X_train, X_valid, y_train, y_valid))

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
# from sklearn.linear_model import RidgeClassifier, SGDClassifier
# from sklearn.naive_bayes import BernoulliNB, GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import LinearSVC, NuSVC, SVC
# from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


# models = []
# # models.append(('LDA', LinearDiscriminantAnalysis()))
# # models.append(('QDA', QuadraticDiscriminantAnalysis()))
# # models.append(('AdaBoost', AdaBoostClassifier()))
# # models.append(('Bagging', BaggingClassifier()))
# # models.append(('Extra Trees Ensemble', ExtraTreesClassifier(n_estimators=1000)))
# # models.append(('Gradient Boosting', GradientBoostingClassifier()))
# # models.append(('Random Forest', RandomForestClassifier(n_estimators=100)))
# # models.append(('Ridge', RidgeClassifier()))
# # models.append(('SGD', SGDClassifier(tol=1e-3, max_iter=10000)))
# # models.append(('BNB', BernoulliNB()))
# # models.append(('GNB', GaussianNB()))
# models.append(('KNN', KNeighborsClassifier()))
# # models.append(('MLP', MLPClassifier()))
# # models.append(('LSVC', LinearSVC(max_iter=100000)))
# models.append(('NuSVC', NuSVC(gamma='scale')))
# # models.append(('SVC', SVC(gamma='scale')))
# # models.append(('DTC', DecisionTreeClassifier()))
# models.append(('ETC', ExtraTreeClassifier()))
# # 
# DECISION_FUNCTIONS = {"Ridge", "SGD", "LSVC", "NuSVC", "SVC"}


# from sklearn.metrics import roc_auc_score, roc_curve
# from sklearn.metrics import matthews_corrcoef
# # %matplotlib inline
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import accuracy_score


# best_model_roc = None
# best_model_roc_name = ""
# best_valid_roc = 0

# # for name, model in models:
# #     model.fit(X_train, y_train)
# #     if name in DECISION_FUNCTIONS:
# #         proba = model.decision_function(X_valid)
# #     else:
# #         try: 
# #             proba = model.decision_function(X_valid)[1:]
# #         except:
# #             proba = model.predict_proba(X_valid)[:, 1]
# #     try:
# #         y_new = y_valid.astype(np.float64)
# #         # print("\n len y_valid: ", len(y_valid))
# #         # print("len proba: ", len(proba), "\n")
# #         # for i in proba:
# #         #     print(type(i))

# #         # print("type y_valid: ", type(y_valid))
# #         # print("type proba: ",type(proba))
# #         roc_score = roc_auc_score(y_new, proba)
# #         accuracy = accuracy_score(y_new, proba)
# #         mcc = matthews_corrcoef(y_new, proba)
# #         precision = precision_score(y_new, proba)
# #         recall = recall_score(y_new, proba)
# #         print("\n\n Model {} \n ROC-score {} \n MCC {} \n Precision {} \n recall {} \n accuracy {}".format(model, roc_score, mcc, precision, recall, accuracy))

# #     except ValueError:
# #         pass





# # 2: 
# # ExtraTreesClassifier(n_estimators=1000)
# # RandomForestClassifier()

# from sklearn.metrics import cohen_kappa_score


# for name, model in models:
#     model.fit(X_train, y_train)
#     if name in DECISION_FUNCTIONS:
#         proba = model.decision_function(X_valid)
#     else:
#         try: 
#             proba = model.decision_function(X_valid)[1:]
#         except:
#             proba = model.predict_proba(X_valid)[:, 1]
#     try:
#         # y_new = y_valid.astype(np.float64)
#         if name in ['KNN', 'MLP', 'NuSVC' , 'SVC']:
#             nproba = proba
#         else: 
#             nproba = proba.astype(np.int64)


#         # print(classification_report(y_valid, nproba, target_names='target'))
#         roc_score = roc_auc_score(y_valid, nproba)
#         # ck = cohen_kappa_score(y_valid, nproba)
#         # print(ck)



#         # print("\n\n------------------------")
#         print("\nModel: ", model)
#         print("ROC: ", roc_score)
#         try: 
#             accuracy = accuracy_score(y_valid, nproba)
#             print("Accuracy: ", accuracy)
#         except:
#             pass
#         try: 
#             mcc = matthews_corrcoef(y_valid, nproba)
#             print("MCC: ", mcc)
#         except:
#             pass
#         try: 
#             precision = precision_score(y_valid, nproba)
#             print("precision: ", precision)
#         except:
#             pass
#         try:
#             recall = recall_score(y_valid, nproba)
#             print("recall: ", recall)
#         except:
#             pass

#         fpr, tpr, _  = roc_curve(y_valid, nproba)
#         plt.figure()
#         plt.plot(fpr, tpr, color='darkorange', label=f"ROC curve (auc = {roc_score})")
#         plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
#         plt.title(f"{name} Results")
#         plt.xlabel("False Positive Rate")
#         plt.ylabel("True Positive Rate")
#         plt.legend(loc="lower right")
#         plt.show()
#         plt.savefig("../model_results/" + str(num_topics) + '_' + str(model) + '.png')
#         if roc_score > best_valid_roc:
#             best_valid_roc = roc_score
#             best_model_roc = model
#             best_model_roc_name = name
#         # print("\n\n Model {} \n ROC-score {} \n MCC {} \n Precision {} \n recall {} \n accuracy {}".format(model, roc_score, mcc, precision, recall, accuracy))

#     except ValueError:
#         pass

# # print(f"Best model is (roc) {best_model_roc_name}")
# # print(f"Best model is (matthews) {best_model_matthews_name}")





# # # # test = pd.read_csv("workfile2.csv", index_col='videoID')[1:]
# # # # submission = pd.read_csv("workfile2.csv", index_col='videoID')[1:]

# # # # test_X = scaler.transform(test)
# # # # if best_model_name in DECISION_FUNCTIONS:
# # # #     submission['top_contestant'] = best_model.decision_function(test_X)
# # # # else:
# # # #     submission['top_contestant'] = best_model.predict_proba(test_X)[:, 1]
# # # # submission.to_csv(f"{best_model_name}_submission.csv")

# # # # print("\n\n\n\n\n\n\n\n\n")