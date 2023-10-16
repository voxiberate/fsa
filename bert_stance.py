#initializing variables:
import time
start_time = time.time()

import warnings
warnings.filterwarnings("ignore")

#getting arguments from command line:
import sys
ask_id=sys.argv[1]

print("VOX AI -stance- Engine starts \n ask_id:",ask_id)

##### db connection ####
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}".format(
            user='postgres',
            password='x9W_nmXKEeTuCs86',
            host="voxiberate-db.cc6fwnibdh50.us-east-1.rds.amazonaws.com",
            database= "develop",
            port=5432
            ))
##################################

#import BERT:
print("importing BERT pipeline..")
from transformers import pipeline

# Load pre-trained model and create pipeline
print("Load pre-trained model and create pipeline..")
classifier = pipeline('sentiment-analysis', model="textattack/bert-base-uncased-imdb")

def classify_stance(ask, comment):
    # Get sentiment of both ask and comment
    ask_sentiment = classifier(ask)[0]['label']
    comment_sentiment = classifier(comment)[0]['label']

    # Compare sentiment: same sentiment implies agreement (0), else disagreement (1)
    if ask_sentiment == comment_sentiment:
        return 0
    else:
        return 1

#load ask:
import pandas as pd
print('reading ask from db..')
query = """ select proposal_content as ask_text from asks where id={}""".format(ask_id)
connection = engine.connect()
ask_db = pd.read_sql(query, connection)
#get the text only:
ask=ask_db.loc[0, "ask_text"]
print("Ask: ",ask)

#load comments:
import pandas as pd
print('reading comments from db..')
query = """ select * from comment where ask_id={}""".format(ask_id)
connection = engine.connect()
comments = pd.read_sql(query, connection)
#comment.head()
print("number of comments:",len(comments))

#should have removed commas! it is still a csv that reads the "body" column after all

sentences = []
for i in range(len(comments)) :
    text=comments.loc[i, "text"]
    #size of comments allowed, by using the number of sentences (".") as threshold:
    #if(text.count('.')>0) :
    sentences.append(text)

print("number of comments with more than N sentences:" , len(sentences))

#assigning each comment to a theme:
comment_theme_df = pd.DataFrame()

#I run the stance classifier here:
#theme 0 is AGREE, 1 for DISAGREE
for i in range(len(sentences)) :
    df2 = {'comment_id': i, 'comment_text': sentences[i], 'theme_id': classify_stance(ask, sentences[i])}
    comment_theme_df = comment_theme_df.append(df2, ignore_index = True)

#preparing data for summarizing:
comment_theme_df_1=comment_theme_df[comment_theme_df['theme_id']==0]
comment_theme_df_2=comment_theme_df[comment_theme_df['theme_id']==1]

temp_sentences_1 = comment_theme_df_1['comment_text'].tolist()
temp_sentences_2 = comment_theme_df_2['comment_text'].tolist()
temp_sentences = [temp_sentences_1, temp_sentences_2]

#needs a clean envinronment (not many runs of summarizing i mean) 
from transformers import pipeline
import os


#import BERT:
print("importing BERT corpus..")
from sentence_transformers import SentenceTransformer

#import several pre-trained models for comparison and assessment (some are large!):
model = SentenceTransformer('distilbert-base-nli-mean-tokens')


## Setting to use the 0th GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Setting to use the bart-large-cnn model for summarization
summarizer = pipeline("summarization", device=0)

print('summarizing theme 0..')
summary_text_1 = summarizer(temp_sentences_1, max_length=1024, min_length=100, do_sample=False)[0]['summary_text']
print('summarizing theme 1..')
summary_text_2 = summarizer(temp_sentences_2, max_length=1024, min_length=100, do_sample=False)[0]['summary_text']

# print(summary_text_1)
# print(summary_text_2)

#merging the theme summaries:
summarized_themes = []

summarized_themes.append(summary_text_1)
summarized_themes.append(summary_text_2)

#store values into a temporary df and then append it to a large one to store into db:
theme_summary_df = pd.DataFrame()

df2 = {'theme_id': 0, 'summary_text': summary_text_1}
theme_summary_df = theme_summary_df.append(df2, ignore_index = True)

df2 = {'theme_id': 1, 'summary_text': summary_text_2}
theme_summary_df = theme_summary_df.append(df2, ignore_index = True)

theme_summary_df

#get sentence embeddings from the ML model:
summarized_themes_embeddings = model.encode(summarized_themes)

#create similarity matrix, initialize at zero:
import numpy as np
similarity_matrix=np.zeros((len(summarized_themes_embeddings), len(summarized_themes_embeddings)))

from sentence_transformers import SentenceTransformer, util
for i in range(len(summarized_themes_embeddings)) :
    for j in range(len(summarized_themes_embeddings)) :

        #calculate similarity (and extract only the float number):
        txt=util.pytorch_cos_sim(summarized_themes_embeddings[i], summarized_themes_embeddings[j])
        similarity_matrix[i,j]=float(txt)

summarized_themes_similarity=similarity_matrix

summarized_themes_similarity=similarity_matrix
summarized_themes_similarity

#create dataframe of array:
summarized_themes_similarity_df = pd.DataFrame(summarized_themes_similarity)
summarized_themes_similarity_df

#get the theme abstract into the comment_theme_df table:
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
query = '''SELECT c.*, t.summary_text, {} as ask_id
            FROM comment_theme_df c
            join theme_summary_df t on c.theme_id=t.theme_id
'''.format(ask_id)
comment_theme_table=pysqldf(query)
comment_theme_table

#get comment and themes embeddings from the ML model:
comment_embeddings = model.encode(comment_theme_table['comment_text'])
summarized_themes_embeddings = model.encode(comment_theme_table['summary_text'])

#create similarity matrix, initialize at zero:
import numpy as np
similarity_comment_theme=[]

from sentence_transformers import SentenceTransformer, util
for i in range(len(comment_theme_table)) :
    #calculate similarity (and extract only the float number):
    txt=util.pytorch_cos_sim(comment_embeddings[i], summarized_themes_embeddings[i])
    print('similarity of',i,'comment to theme',float(txt))
    similarity_comment_theme.append(round(float(txt),3))

comment_theme_table['similarity']=similarity_comment_theme

#rank the table according to comments similarity on themes:
query = '''SELECT *
            FROM comment_theme_table
            order by theme_id, similarity desc
'''
comment_theme_table=pysqldf(query)

comment_theme_table['ranking'] = comment_theme_table.groupby('theme_id')['similarity'].rank(ascending=False)
#convert to integer:
comment_theme_table['ranking']=comment_theme_table['ranking'].astype(int)

comment_theme_table.to_csv('comment_theme_ranking.csv',index=False)

print(len(comments),'comments, ',len(sentences),'that are long enough, \n',
      len(comment_theme_table['comment_text'].unique()),' comments that are used into theme creation, \n',
      len(comment_theme_table['theme_id'].unique()),' themes | ask_id:', ask_id, '\n',
      comment_theme_table['theme_id'].value_counts().loc[0], ' and ', comment_theme_table['theme_id'].value_counts().loc[1], ' comments that agree / disagree')

############## FINALIZING WRITING, COMMENTS, THEMES AND SIMILARITIES ##############
print('truncating comment_theme_ranking table before write..')
from sqlalchemy.sql import text as sa_text
connection.execute(sa_text('''TRUNCATE TABLE comment_theme_ranking_complete''').execution_options(autocommit=True))
print('done truncating!')

#write:
import io
print('attempting to write to db table..')

#fixing datatypes of freaking python..:
comment_theme_table = comment_theme_table.astype({'comment_id':'int'})
comment_theme_table = comment_theme_table.astype({'theme_id':'int'})

data_to_write=comment_theme_table

conn = engine.raw_connection()
cur = conn.cursor()
output = io.StringIO()
data_to_write.to_csv(output, sep='\t', header=False, index=False)
output.seek(0)
contents = output.getvalue()
cur.copy_from(output, 'comment_theme_ranking_complete', null="") # null values become ''
conn.commit()
print('writing complete:',len(data_to_write),'rows on final table')

###############################
##write theme similarity table:
print('write theme similarity table, custom size!..')

#write:
import io
import pickle

#uses the connection made above:
conn.set_session(autocommit=True)

cur = conn.cursor() #to make sure autocommit=true
cur.execute(
    """
    DROP TABLE IF EXISTS numpy_arrays;
    CREATE TABLE numpy_arrays (
        uuid VARCHAR PRIMARY KEY,
        np_array_bytes BYTEA
    )
    """
)
some_array_uuid = 'summarized_themes_similarity'

cur.execute(
    """
    INSERT INTO numpy_arrays(uuid, np_array_bytes)
    VALUES (%s, %s)
    """,
    (some_array_uuid, pickle.dumps(summarized_themes_similarity))
)
print('done writing custom size numpy array')

print('--- VOXIBERATE AI completed ---')

print("--- %s seconds ---" % round((time.time() - start_time),0))
