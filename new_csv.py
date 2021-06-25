
import pandas as pd
import csv
import numpy as np

doc = pd.read_csv('../Data/new_contestants.csv', error_bad_lines=False)
doc.head()


# open input CSV file as source
# open output CSV file as result
with open('../Data/new_contestants.csv', "r") as source:
    reader = csv.reader(source)
      
    with open("output.csv", "w") as result:
        writer = csv.writer(result)
    
        i = 0
        for r in reader:
            if i ==0:
                writer.writerow((r[0], r[1], "top_contestant", "videoID"))
            i += 1
            # Use CSV Index to remove a column from CSV
            #r[3] = r['year']
            if i > 1156:
                nl = r[19][28:]
                if r[8] != "NA":
                    if int(r[8]) < 6:
                        writer.writerow((r[0], r[1], 1, nl))
                else:
                    writer.writerow((r[0], r[1], 0, nl))




newdoc = pd.read_csv('output.csv', error_bad_lines=False)

                   