CLIENT_SECRETS_FILE = "client_secret.json"

SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

import os
import pickle
import google.oauth2.credentials
import pandas
import csv

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import get_links


api_key = 'AIzaSyAKqWZmrZApkh92SmbUIKUxppErLT7UreA'

filename = "../data/contestants.csv"
f = pandas.read_csv(filename)

year = 2015

def get_authenticated_service():
    credentials = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            credentials = pickle.load(token)
    #  Check if the credentials are invalid or do not exist
    if not credentials or not credentials.valid:
        # Check if the credentials have expired
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            credentials = flow.run_console()
 
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(credentials, token)
 
    return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)


if __name__ == '__main__':
    # When running locally, disable OAuthlib's HTTPs verification. When
    # running in production *do not* leave this option enabled.
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    service = get_authenticated_service()



  

def video_comments(video_id):
    # empty list for storing reply
    replies = []
  
    # creating youtube resource object
    youtube = build('youtube', 'v3',
                    developerKey=api_key)
  
    # retrieve youtube video results
    video_response=youtube.commentThreads().list(
    part='snippet,replies',
    maxResults=100,
    videoId=video_id
    ).execute()
  
    long_list = []
    
    for item in video_response['items']:
        # Extracting comments
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        long_list.append(comment)

    return long_list


# todo: schrijf comments per jaar in een file -> gebeurt in get_links
def write_to_csv(video_id, long_list, year):
    filename = str(year) + "_comments_" + str(video_id) + ".csv"
    with open(filename, 'w') as comments_file:
        comments_writer = csv.writer(comments_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        comments_writer.writerow(['Video ID', 'Comment'])
        for comment in long_list:
            comments_writer.writerow([video_id, comment])


all_video_ids = get_links.get_list_with_ids_per_year(year)
# some_video_ids = all_video_ids[350:]

# waardoor wordt de error veroorzaakt? Is het de requests per tijd? Dan zou een sleep() kunnen helpen
for video_id in all_video_ids:
    try:
        comments = video_comments(video_id)
        write_to_csv(video_id, comments, year)
    except:
        continue
