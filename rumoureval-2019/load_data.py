import json
import os
import fnmatch
import pandas as pd
import re
import numpy as np
import string

def main():
    df_train = create_df_data('twitter-english', 'train-key.json')
    df_eval = create_df_data('twitter-english', 'dev-key.json')
    df_test = create_df_data('twitter-en-test-data', 'final-eval-key.json')
    
    #cleaning data
    df_train['text'] = df_train['text'].apply(lambda x: clean_data(x))
    df_eval['text'] = df_eval['text'].apply(lambda x: clean_data(x))
    df_test['text'] = df_test['text'].apply(lambda x: clean_data(x))
    
    create_csv_files(df_train, df_test, df_eval)
    

def create_df_data(path_training, path_answer):
    s1 = pd.Series([0,'test', 0])
    df = pd.DataFrame([list(s1)], columns=['id', 'text','label'])
    index = 0

    for path, dirs, files in os.walk(path_training):
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
    df_label = pd.read_json(path_answer)
    first_col = df_label.columns[0]
    df_label.rename(columns={first_col:'label'}, inplace=True)
    
    #add labels
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

def clean_data(text):
    #remove links
    text = re.sub(r'http\S+', '', text)

    #remove punct except @
    punc_to_remove = string.punctuation
    punc_to_remove.replace('@', '')
    text = "".join([char for char in text if char not in punc_to_remove])

    #lower case
    return text.lower()


def create_csv_files(df_train, df_test, df_eval):
    df_train.to_csv('train_data.csv', index=False)
    df_test.to_csv('test_data.csv', index=False)
    df_eval.to_csv('eval_data.csv', index=False)


if __name__ == "__main__":
    main()