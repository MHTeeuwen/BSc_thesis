import pandas as pd
import csv
import os
import re
from googletrans import Translator
translator = Translator()

FILENAME = "translated_comments.csv"
NEWFILENAME = 'translated_comments_3.csv'

data = pd.read_csv(FILENAME)



# with open(FILENAME) as inf, open(NEWFILENAME, 'w') as outf:
#     reader = csv.reader(inf)
#     writer = csv.writer(outf)
#     for line in reader:
#         newline = translator.translate(line[2]).text # translate comment to english

#         newnewline = re.sub(r'[^\w\s]','', newline) # remove punctuation
#         newnewnewline = newnewline.lower() # convert to lower case


#         writer.writerow([line[0], line[1], newnewnewline])


# comment = "...It's a comment!!1!  :-)"

# new_comment = re.sub(r'[^\w\s]','', comment) 
# new_comment = new_comment.lower()

# print(new_comment)


path = "../Data/2015_comments/"
file = 'combined.csv'
FILENAME = path + file


def translate_csv(filename):
    new_filename = path + "translated_" + file
    # new_filename = NEWFILENAME
    with open(filename) as inf, open(new_filename, 'w') as outf:
        reader = csv.reader(inf)
        writer = csv.writer(outf)

        # TODO: <br> <href> etc. alleen de << >> worden weggehaald. Hierdoor komt er br aan een wordt toegevoegd.
        for line in reader:
            
            # TODO: Test of deze translator werkt. Ik zie nml nog russische tekens staan. Misschien kunnen deze tekens niet geinterpreteerd worden? 
            newline = translator.translate(line[1]).text # translate comment to english

            newnewline = re.sub(r'[^\w\s]','', newline) # remove punctuation
            newnewnewline = newnewline.lower() # convert to lower case

            writer.writerow([line[0], newnewnewline])

translate_csv(FILENAME)

# translate_csv(FILENAME2)



# schrijf in thesis over russicsche tekens