from gensim import corpora, models
from nltk.tokenize import word_tokenize

def perform_topic_modeling(texts):
    # Tokenize the text and create a dictionary
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    
    # Convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    # Perform Latent Dirichlet Allocation (LDA) for topic modeling
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
    
    # Print the topics
    topics = lda_model.print_topics(num_words=5)
    for topic in topics:
        print(topic)


perform_topic_modeling(texts)
