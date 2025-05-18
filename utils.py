import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from transformers import BertTokenizer
import nltk

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def parse_chat(filepath):
    user=[]
    ai=[]
    with open(filepath, 'r', encoding = 'utf-8') as file:
        content = file.read()
    
    pattern = re.findall(r'(User|AI):\s*(.*?)(?=(?:User|AI):|\Z)', content, re.DOTALL)
    
    for speaker, message in pattern:
        cleaned = message.strip()
        if speaker == 'User':
            user.append(cleaned)
        elif speaker == "AI":
            ai.append(cleaned)
    return user, ai

def msg_stats(user, ai):
    return{
        "Total": len(user) + len (ai),
        "User": len(user),
        "AI": len(ai)
    }

def bert_tokenize(text):
    tokens = tokenizer.tokenize(text.lower())
    return [t for t in tokens if t not in STOPWORDS and re.match(r'[a-z]+', t)]

def tfidf_key(messages, top_n=5):
    processed = [" ".join(bert_tokenize(msg)) for msg in messages]
    vectonizer = TfidfVectorizer()
    X = vectonizer.fit_transform(processed)
    scores = X.sum(axis=0).A1
    vocab = vectonizer.get_feature_names_out()
    keyword_scores = list(zip(vocab, scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    return [word for word, _ in keyword_scores[:top_n]]

def generate_summary(stats, keywords):
    return (
        f"\nSummary:\n"
        f"- Total exchanges: {stats['Total']}\n"
        f"- User messages: {stats['User']}\n"
        f"- AI messages: {stats['AI']}\n"
        f"- Top TF-IDF keywords: {', '.join(keywords)}\n"
    )

def plot_msg_stats(stats, filename=None):
    sns.set(style="whitegrid")
    data = {'Speaker': ['User', 'AI'], 'Messages': [stats['User'], stats['AI']]}
    plt.figure(figsize=(6, 4))
    sns.barplot(x='Speaker', y='Messages', data=data)
    plt.title('Message Count by Speaker')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show(block=False)
    plt.pause(10)
    plt.close()
