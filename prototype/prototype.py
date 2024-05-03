import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def read_csv(file_path):
    # define column names since the CSV doesn't have headers
    column_names = ['title', 'link', 'content']
    # read the CSV with no header and assign column names
    data = pd.read_csv(file_path, header=None, names=column_names)
    return data

def clean_content(content):
    # find the index of the token '♦' and slice the content up to that point or until "\n\n"
    end_index = content.find('♦')
    if end_index != -1:
        # check if there is a double newline after the token, and slice accordingly
        double_newline_index = content.find('\n\n', end_index)
        if double_newline_index != -1:
            content = content[:end_index] + content[double_newline_index+2:]
        else:
            content = content[:end_index]
    return content

def preprocess(text):
    # convert text to lowercase
    text = clean_content(text)
  
    text = text.lower()

    # Manually replace incorrect newline representation
    text = text.replace('\\n', '\n')


    # Replace newline characters with a single space
    text = re.sub(r'\n', ' ', text)
    
    # Replace other excessive whitespaces (including tabs and multiple spaces) with a single space
    text = re.sub(r'\s+', ' ', text)

    # remove all non-alphanumeric characters except spaces
    text = re.sub(r'[^\w\s]', '', text)
    
    # tokenize text by splitting into words
    words = text.split()
    
    # join the processed words back into a string
    return ' '.join(words)

def calculate_similarity(question, contents):
    # using TF-IDF and cosine similarity to calculate similarity
    vectorizer = TfidfVectorizer()
    text_data = [question] + contents
    tfidf_matrix = vectorizer.fit_transform(text_data)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return cosine_similarities.flatten()

def find_most_similar_content(question, data):
    # preprocess the question and the content
    processed_question = preprocess(question)
    processed_contents = [preprocess(content) for content in data['content']]

    # calculate similarities
    similarities = calculate_similarity(processed_question, processed_contents)
    
    # get the index of the most similar content
    most_similar_index = similarities.argmax()
    
    #return processed_contents[most_similar_index], data['content'][most_similar_index] #may need to comment this out
    return  data['content'][most_similar_index] #chatgpt might understand this better

def findTopic(question):
    file_path = '../eldenRingWikiText.csv'
    data = read_csv(file_path)

    #return find_most_similar_content(question, data)[1]
    return find_most_similar_content(question, data)


def main():
    # file path for the CSV
    file_path = '../eldenRingWikiText.csv'
    # example question to find similar content
    question = "what are all the weapon ctypes"

    # read and process the CSV data
    data = read_csv(file_path)
    
    # find the most similar content
    #most_similar_content = find_most_similar_content(question, data)[0] # comment out
    #raw_most_similar_content = find_most_similar_content(question, data)[1] #comment out
    raw_most_similar_content = find_most_similar_content(question, data) # I would use this


    #print(most_similar_content) #comment this out
    #print("========================================================")
    print(raw_most_similar_content) #I would have chatgpt look at this instead. it might not correctly understand the preprocessed version


if __name__ == "__main__":
    main()
