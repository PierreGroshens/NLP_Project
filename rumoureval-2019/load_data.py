import json
import os
import fnmatch
import pandas as pd
import re
import numpy as np
import string


#for cleaning
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk import WordNetLemmatizer

def main():
    df_train = create_df_data('twitter-english', 'train-key.json')
    df_eval = create_df_data('twitter-english', 'dev-key.json')
    df_test = create_df_data('twitter-en-test-data', 'final-eval-key.json')
    print("\n df created \n")
    #cleaning data
    
    global EMOJI_PATTERN
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+"
        )


    global stop_words_list
    stop_words_list = stopwords.words('english')
    
    #cleaning can be modified by updating args given to clean_data function
    df_train['text'] = df_train['text'].apply(lambda x: clean_data(x, lowercase=True))
    df_eval['text'] = df_eval['text'].apply(lambda x: clean_data(x, lowercase=True))
    df_test['text'] = df_test['text'].apply(lambda x: clean_data(x, lowercase=True))
    print("\n df cleaned \n")
    
    global full_concatenation 
    full_concatenation = True
    
    #if you want to save full history dataset then True 
    #if you want to save chain dataset then False
    
    df_train = create_history(df_train, 'twitter-english')
    df_eval = create_history(df_eval, 'twitter-english')
    df_test = create_history(df_test, 'twitter-en-test-data')
    
    print("\n df concatenated \n")

    create_csv_files(df_train, df_test, df_eval)
    

def create_df_data(path_training, path_answer):
    s1 = pd.Series([0,'test', 0])
    df = pd.DataFrame([list(s1)], columns=['id', 'text','label'])
    index = 0

    for path, dirs, files in os.walk(path_training):
        #we are only interested in source-tweet folder
        #this is where data are
        #we then open the json file and take the relevant information
        if (path.endswith('source-tweet') or path.endswith('replies')):
            for filename in os.listdir(path):
                if filename.endswith('.json'):
                    with open(os.path.join(path,filename)) as jsonFile:
                        jsonSourceTweet = json.load(jsonFile)

                        text_tweet = jsonSourceTweet['text']
                        id_tweet = int(re.sub("[^0-9]", "", filename))

                        newline = [id_tweet, text_tweet, index]
                        df.loc[index] = newline

                        index += 1

    #rename column into label
    path_answer = os.path.realpath(path_answer)                   
    df_label = pd.read_json(path_answer)
    first_col = df_label.columns[0]
    df_label.rename(columns={first_col:'label'}, inplace=True)
    
    #add labels to df object
    #labels not found are replaced by not_found
    #not_found labels are tweets that don't appear in the training set but testing or evaluating one
    for i in range(len(df)):
        try:
            index = df.loc[i].id
            df.loc[i, 'label'] = df_label.loc[str(index)].label
        except KeyError:
            df.loc[i, 'label'] = 'not_found'

    #delete unecessary labels
    index_to_delete = df[df.label == "not_found"].index
    mask = np.logical_not(df.index.isin(index_to_delete))
    df_final = df[mask]


    return df_final

def clean_data(text, punctuation = True, keep_stop_words = True, lemmatize = False, keep_emojis = True, lowercase = False, keep_links = True):
    #remove links
    if (keep_links == False):
        text = re.sub(r'http\S+', ' <LINK> ', text)

    if (keep_emojis == False):
        #emojis out
        text = re.sub(EMOJI_PATTERN, ' <EMOJIS> ', text)        
        
    if (punctuation == False):
        
        #remove punct except @
        punc_to_remove = string.punctuation.replace('@', '')
        text = "".join([char for char in text if char not in punc_to_remove])
        
    if (keep_stop_words == False):
        text = remove_stop_words(text)        
        
    if (lemmatize == True):
        text = lemmatize_words(text)        
    
    
    #lower case
    if(lowercase == True):
        text = text.lower()
        
    return text

def remove_stop_words(text):
    tokens = word_tokenize(text)
    tweet_tokens_wst = [word for word in tokens if word not in stop_words_list]
    tweet_tokens_wst = TreebankWordDetokenizer().detokenize(tweet_tokens_wst)
    return tweet_tokens_wst

def lemmatize_words(text):
    
    wn = WordNetLemmatizer()
    #turn tweet into tokens to lemmatize them
    tokens = word_tokenize(text)

    tweet_lemmatized = [wn.lemmatize(token) for token in tokens]
    tweet_lemmatized = TreebankWordDetokenizer().detokenize(tweet_lemmatized)
    return tweet_lemmatized

def create_csv_files(df_train, df_test, df_eval):
    df_train.to_csv('train_data.csv', index=False)
    df_test.to_csv('test_data.csv', index=False)
    df_eval.to_csv('eval_data.csv', index=False)

def create_history(df, path):
    #create new column for parent id of tweet
    df.insert(1, "parent_id", None)
    
    fill_parent_id(df, path)
    df_final = df.copy()
    concatenate(df, df_final)
    
    return df_final if full_concatenation == False else df

def fill_parent_id(df, path):
    #add parent id col to dataframe object
    
    #we will go look in the repliees.json file to save parent id
    for path, dirs, files in os.walk(path):

        if (path.endswith("replies")):

            for _, _, files in os.walk(path):
                for file in files:

                    with open(os.path.join(path,file)) as jsonFile:
                        data = json.load(jsonFile)
                        parent_id = data["in_reply_to_status_id"]
                        tweet_id = re.sub('\D*', '', file)

                        df.loc[df.id == int(tweet_id), "parent_id"] = parent_id

def concatenate(df, df_final):
    for i in df.index:

        id_to_search = df.loc[i].parent_id

        if (pd.isna(id_to_search) == False):
            try:
                context = df[df.id == id_to_search].text.values[0]

                end_sentence = df.loc[i].text
                #if full concatenation or not, then not same df object modified
                df_modified = df if full_concatenation == True else df_final
                df_modified.loc[i, 'text'] = context + ' <REPLY> ' + end_sentence
            
            #some parent tweets do not exist in all the dataset. 
            #this happens 3 times
            except IndexError:
                print('warning')
                print(id_to_search)
    
if __name__ == "__main__":
    main()