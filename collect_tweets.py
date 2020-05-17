from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from subprocess import STDOUT
import time
import json

#Variables that contains the user credentials to access Twitter API (twitter keys hidden)
access_token = ""
access_token_secret = ""
consumer_key = ""
consumer_secret = ""

class Listener(StreamListener):
    def on_data(self,data):
        try:
            print (data)
            saveFile=open('tweetscollect.csv','a')
            saveFile.write(data)
            saveFile.write('\n')
            saveFile.close()
            return True
        except BaseException as e:
            print ('failed ondata,'),str(e)
            time.sleep(5)

    def on_error(self,status):
        print (status)
    
    if __name__ == '__main__':
        l = Listener ()
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        twitterStream = Stream(auth,l)
        
        #This line filter Twitter Streams to capture data by the keywords:
        twitterStream.filter(track=['Apple’])