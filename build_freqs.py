def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs
  
  
  
  ### explanation of frequencies to do feature extraction
  '''
import pandas as pd
df=pd.DataFrame({'tweets':["i am happy because i am learning nlp happy","i hated that movie","i am sad because i am not happy","i love working at dl"],"label":[1,0,0,1]})
df_processed=df
df_processed["tweets"]=df_processed["tweets"].apply(process_tweet)
df_processed

tweets	label
0	[happi, learn, nlp, happi]	1
1	[hate, movi]	0
2	[sad, happi]	0
3	[love, work, dl]	1

tweets=df_processed['tweets']
ys=df_processed['label']
yslist = np.squeeze(ys).tolist()

# Start with an empty dictionary and populate it by looping over all tweets
# and over all processed words in each tweet.
freqs = {}
for y,tweet in zip(yslist,tweets):
    for word in tweet:
        pair = (word, y)
        if pair in freqs:
            freqs[pair] += 1
        else:
            freqs[pair] = 1
            
      
      
freqs

{('happi', 1): 2,
 ('learn', 1): 1,
 ('nlp', 1): 1,
 ('hate', 0): 1,
 ('movi', 0): 1,
 ('sad', 0): 1,
 ('happi', 0): 1,
 ('love', 1): 1,
 ('work', 1): 1,
 ('dl', 1): 1}
 
 '''
  
  
  
  
  
  
  
  
  
  
  
  
