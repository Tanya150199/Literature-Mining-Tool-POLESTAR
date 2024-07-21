import requests
from xml.etree import ElementTree as ET
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from collections import Counter

# Initialize NLP pipelines
summarizer = pipeline("summarization")
ner_pipeline = pipeline("ner")
qa_pipeline = pipeline("question-answering")

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

def advanced_preprocess(texts, use_ngrams=True, custom_stopwords=None):
    lemmatizer = WordNetLemmatizer()
    if custom_stopwords is None:
        custom_stopwords = set(stopwords.words('english'))
    processed_texts = []
    for text in texts:
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in custom_stopwords and word.isalpha()]
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
        if use_ngrams:
            bigrams = [' '.join(bigram) for bigram in nltk.bigrams(tokens)]
            lemmatized += bigrams
        processed_texts.append(' '.join(lemmatized))
    return processed_texts


def summarize_text(abstract_text):
    if abstract_text:
        length_of_abstract = len(abstract_text.split())
        summary = summarizer(abstract_text, max_length=min(130, length_of_abstract + 10), min_length=30, do_sample=False)[0]['summary_text']
        return summary
    return 'No abstract available'

def perform_topic_modeling(texts, n_topics=5):
    processed_texts = advanced_preprocess(texts)
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, stop_words='english', ngram_range=(1, 2))
    dtm = vectorizer.fit_transform(processed_texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(dtm)
    return lda, vectorizer


def extract_keywords(ner_results):
    return [kw['word'] for kw in ner_results if len(kw['word']) > 1]


def fetch_biopolymer_articles(query, max_results):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        'db': 'pubmed',
        'term': query,
        'retmax': max_results,
        'usehistory': 'y',
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        id_list = [id_tag.text for id_tag in root.findall('./IdList/Id')]
        return id_list
    else:
        print("Failed to fetch data:", response.status_code)
        return []


def fetch_article_details(article_ids):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        'db': 'pubmed',
        'id': ','.join(article_ids),
        'retmode': 'xml',
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.text
    else:
        print("Failed to fetch article details")
        return None


def parse_articles(xml_data):
    try:
        root = ET.fromstring(xml_data)
        articles = []
        for article in root.findall('.//PubmedArticle'):
            title = article.find('.//ArticleTitle').text
            abstract_text_element = article.find('.//Abstract/AbstractText')
            abstract_text = abstract_text_element.text if abstract_text_element is not None else ''
            pmid_element = article.find('.//PMID')
            pmid = pmid_element.text if pmid_element is not None else ''
            paper_url = f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'
            summary = summarizer(abstract_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] if abstract_text else 'No abstract available'
            keywords = ner_pipeline(abstract_text)
            keyword_list = extract_keywords(keywords) if abstract_text else []
            articles.append({
                'title': title,
                'abstract': abstract_text,
                'summary': summary,
                'url': paper_url
            })
        return articles
    except ET.ParseError as e:
        print(f"An error occurred while parsing XML: {e}")
        return []


def get_distinct_topic_keywords(lda_model, vectorizer, n_top_words=10):
    topic_keywords = []
    for topic_idx, topic in enumerate(lda_model.components_):
        # Get the top n keywords for this topic
        top_keywords_idx = topic.argsort()[-n_top_words:]
        topic_keywords.append([vectorizer.get_feature_names_out()[i] for i in top_keywords_idx])
    
    # ensure the keywords are unique across topics by filtering out repeated keywords
    unique_keywords = set()
    distinct_topic_keywords = []
    for keywords in topic_keywords:
        distinct_keywords = [word for word in keywords if word not in unique_keywords]
        unique_keywords.update(distinct_keywords)
        distinct_topic_keywords.append(distinct_keywords)
    return distinct_topic_keywords

# Function to fetch articles from DOAJ
def fetch_doaj_articles(query, max_results=10):
    base_url = f"https://doaj.org/api/v2/search/articles/{query}"
    params = {
        'pageSize': max_results
    }
    response = requests.get(base_url, params=params)
    articles = []
    
    if response.status_code == 200:
        articles_data = response.json()
        for article in articles_data.get('results', []):
            title = article['bibjson'].get('title')
            abstract = article['bibjson'].get('abstract', 'No abstract provided')
            link = article['bibjson']['link'][0]['url']  # Assuming the first link is to the article
            summary = summarize_text(abstract)
            articles.append({
                'title': title,
                'abstract': abstract,
                'summary': summary,
                'url': link
            })
        return articles
    else:
        print("Failed to fetch articles from DOAJ")
        return []


def literature_mining_tool(query, max_results=10, source='pubmed'):
    if source == 'doaj':
        articles = fetch_doaj_articles(query, max_results)
    elif source == 'pubmed':
        article_ids = fetch_biopolymer_articles(query, max_results)
        if not article_ids:
            print("No articles found for the query on PubMed.")
            return []
        article_details_xml = fetch_article_details(article_ids)
        if not article_details_xml:
            print("Failed to fetch article details from PubMed.")
            return []
        articles = parse_articles(article_details_xml)
    else:
        print("Source not recognized.")
        return []
    
    if not articles:
        print("No articles to parse.")
        return []
    
    abstract_texts = [article['abstract'] for article in articles]
    lda_model, vectorizer = perform_topic_modeling(abstract_texts, n_topics=5)
    distinct_topic_keywords = get_distinct_topic_keywords(lda_model, vectorizer)
    
    for article in articles:
        abstract = article['abstract']
        if abstract:
            topic_distribution = lda_model.transform(vectorizer.transform([abstract]))[0]
            dominant_topic = topic_distribution.argmax()
            article['dominant_topic'] = "Topic {}: {}".format(dominant_topic, ', '.join(distinct_topic_keywords[dominant_topic]))
            article['summary'] = summarize_text(abstract)

    df = pd.DataFrame(articles)
    output_file = 'literature_mining_results.csv'
    df.to_csv(output_file, index=False)
    print(f"Results have been saved to {output_file}")
    return df


def literature_mining_tool(query, max_results=10, source='pubmed'):
    if source == 'doaj':
        articles = fetch_doaj_articles(query, max_results)
    elif source == 'pubmed':
        article_ids = fetch_biopolymer_articles(query, max_results)
        if not article_ids:
            print("No articles found for the query on PubMed.")
            return []
        article_details_xml = fetch_article_details(article_ids)
        if not article_details_xml:
            print("Failed to fetch article details from PubMed.")
            return []
        articles = parse_articles(article_details_xml)
    else:
        print("Source not recognized.")
        return []
    
    if not articles:
        print("No articles to parse.")
        return []
    
    abstract_texts = [article['abstract'] for article in articles]
    lda_model, vectorizer = perform_topic_modeling(abstract_texts, n_topics=5)
    distinct_topic_keywords = get_distinct_topic_keywords(lda_model, vectorizer)
    
    for article in articles:
        abstract = article['abstract']
        if abstract:
            topic_distribution = lda_model.transform(vectorizer.transform([abstract]))[0]
            dominant_topic = topic_distribution.argmax()
            article['dominant_topic'] = "Topic {}: {}".format(dominant_topic, ', '.join(distinct_topic_keywords[dominant_topic]))
            article['summary'] = summarize_text(abstract)
            
    df = pd.DataFrame(articles)
    output_file = 'literature_mining_results.csv'
    df.to_csv(output_file, index=False)
    print(f"Results have been saved to {output_file}")
    return df

def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return {
        'answer': result['answer'],
        'score': result['score']
    }


