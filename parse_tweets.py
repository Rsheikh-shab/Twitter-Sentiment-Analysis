import json

with open('tweetscollect.csv', 'r') as data_file:
    count=0
    #def on_data(self, data):
    for line in data_file:
        try:
            jsonData = json.loads(line)
            createdAt = jsonData['created_at']
            text = jsonData['text']
            lang=jsonData['lang']
            loc=jsonData['user']['location']
            if ('RT @' not in text) and ('Retweeted' not in text):
                saveFile=open('parsed_ML_tweets.csv','a')
                saveFile.write("text: "+ text+'\n')
                saveFile.write('\n')
                saveFile.close()
                #return True
                count=count+1
        except Exception:
            pass
    print (count)