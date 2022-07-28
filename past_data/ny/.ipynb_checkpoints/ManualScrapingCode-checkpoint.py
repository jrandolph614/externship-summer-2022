#intro scraping code
#not automated

import pandas as pd
import tweepy

consumer_key = "B0DcFInjpYIrFIFoTfLDvM0hi"
consumer_secret = "P133ySqrOiiKJ0Jc44UPYeAJHooBSsrrpDliHWGNg29YI1MmBq"

access_key = "2185187565-mrF3ibnU2gYYgAXb30PYkVaxkZeapv1Rz0tQb4n"
access_secret= "2uQbB5CGpVo6WRfQPl49YUJeOOVSnFLnLQJn4Y7a7LGXb"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

#q is the keyword you are scraping for; you can include multiple here like q = ["Albany","daddy"]
#indicate the date since and number of tweets you aim to scrape here as well
q = ["Albany"]
date_since = "2022-01-01"
numtweet = 1000

def printtweetdata(n, ith_tweet):
        print()
        print(f"Tweet {n}:")
        print(f"Username:{ith_tweet[0]}")
        print(f"Description:{ith_tweet[1]}")
        print(f"Location:{ith_tweet[2]}")
        print(f"Following Count:{ith_tweet[3]}")
        print(f"Follower Count:{ith_tweet[4]}")
        print(f"Total Tweets:{ith_tweet[5]}")
        print(f"Retweet Count:{ith_tweet[6]}")
        print(f"Tweet Text:{ith_tweet[7]}")
        print(f"Hashtags Used:{ith_tweet[8]}")

def scrape(q, date_since, numtweet):
        db = pd.DataFrame(columns=['username',
                                   'description',
                                   'location',
                                   'following',
                                   'followers',
                                   'totaltweets',
                                   'retweetcount',
                                   'text',
                                   'hashtags', 
                                   'date'])
 
        # We are using .Cursor() to search
        # through twitter for the required tweets.
        # The number of tweets can be
        # restricted using .items(number of tweets)
        tweets = tweepy.Cursor(api.search_tweets,
                               q, lang="en",
                               since_id=date_since,
                               tweet_mode='extended').items(numtweet)
 
 
        # .Cursor() returns an iterable object. Each item in
        # the iterator has various attributes
        # that you can access to
        # get information about each tweet
        list_tweets = [tweet for tweet in tweets]
 
        # Counter to maintain Tweet Count
        i = 1
 
        # we will iterate over each tweet in the
        # list for extracting information about each tweet
        for tweet in list_tweets:
                username = tweet.user.screen_name
                description = tweet.user.description
                location = tweet.user.location
                following = tweet.user.friends_count
                followers = tweet.user.followers_count
                totaltweets = tweet.user.statuses_count
                retweetcount = tweet.retweet_count
                hashtags = tweet.entities['hashtags']
                date = tweet.created_at
 
                # Retweets can be distinguished by
                # a retweeted_status attribute,
                # in case it is an invalid reference,
                # except block will be executed
                try:
                        text = tweet.retweeted_status.full_text
                except AttributeError:
                        text = tweet.full_text
                hashtext = list()
                for j in range(0, len(hashtags)):
                        hashtext.append(hashtags[j]['text'])
 
                # Here we are appending all the
                # extracted information in the DataFrame
                ith_tweet = [username, description,
                             location, following,
                             followers, totaltweets,
                             retweetcount, text, hashtext, date]
                db.loc[len(db)] = ith_tweet
 
                # Function call to print tweet data on screen
                #printtweetdata(i, ith_tweet)
                i = i+1
                
         #change the filename each time you scrape so you don't rewrite the last file      
        filename = r'/Users/saraosowski/Documents/tweets_albanydaddy.csv'
 
        db.to_csv(filename)

scrape(q, date_since, numtweet)

