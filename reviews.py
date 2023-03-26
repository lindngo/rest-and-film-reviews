## Text Representation of Restaurant and Film Reviews
import nltk
nltk.download('punkt') 
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger') 

import pandas as pd
import gensim
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Using header = None to prevent first row being the header
data = pd.read_csv("reviews_data.csv", header = None)

# Grab only the second column
data_review = data.iloc[:,1]

# Adding each row into a list
data_review = data_review.values.tolist()
data_review

# 1. Tokenize, 2. Lemmatize, 3. Remove stop words and punctuation

processed_data = []

for x in data_review:
    token = nltk.word_tokenize(x)
    #print(token)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_token = [lemmatizer.lemmatize(x) for x in token if x.isalpha()]
    #print(lemmatized_token)
    stop_words_removed = [x for x in lemmatized_token if not x in stopwords.words('english') if x.isalpha()]
    processed_data.append(stop_words_removed)

processed_data

processed_data2 = list(map(' '.join, processed_data))
processed_data2

#4. TD-IDF vectors (min doc frequency = 3; 2-gram)

vectorizer = TfidfVectorizer(ngram_range = (1, 2), min_df = 3)
vectorizer.fit(processed_data2)
print(vectorizer.vocabulary_)

v2 = vectorizer.transform(processed_data2)
print(v2.toarray())

# 4. Vector Dimensions
tfidf_matrix = vectorizer.transform(processed_data2)
print("Dimensions:", tfidf_matrix.shape)

# Generating final TF-IDF vectors CSV
import pandas as pd
df = pd.DataFrame(v2.toarray(), columns = vectorizer.get_feature_names_out())
df.insert(loc = 0, column = "Reviews", value = processed_data2)
df.to_csv("tf-idf-vectors.csv", index = False)

# 5. POS-tag and TD-IDF (min doc frequency = 4)

POS_data_review = []

for x in data_review:
    token = nltk.word_tokenize(x)
    POS_token = nltk.pos_tag(token)
    #print(POS_token)
    POS_token_temp = []
    
    for i in POS_token:
        POS_token_temp.append(i[0] + "_" + i[1])
    
    POS_data_review.append(" ".join(POS_token_temp))
    
print(POS_data_review)

vectorizer = TfidfVectorizer(min_df = 4)
vectorizer.fit(POS_data_review)
print(vectorizer.vocabulary_)

POS_v2 = vectorizer.transform(POS_data_review)
print(POS_v2.toarray())

# 5. Vector Dimensions
tfidf_matrix = vectorizer.transform(POS_data_review)
print("Dimensions:", tfidf_matrix.shape)

# Generating POS‐tag TF‐IDF vectors CSV
import pandas as pd
df = pd.DataFrame(POS_v2.toarray(), columns = vectorizer.get_feature_names_out())
df.insert(loc = 0, column = "Reviews", value = data_review)
df.to_csv("post-tag.csv", index = False)