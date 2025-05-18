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
    """
    Parses a chat log file and separates messages by 'User' and 'AI'.

    Args:
        filepath (str): Path to the chat log file.

    Returns:
        user (list): List of messages from the user.
        ai (list): List of messages from the AI.
    """

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
     """
    Computes message statistics for user and AI.

    Args:
        user (list): User messages.
        ai (list): AI messages.

    Returns:
        dict: Dictionary containing counts of messages.
    """
     return{
        "Total": len(user) + len (ai),
        "User": len(user),
        "AI": len(ai)
     }

def bert_tokenize(text):
    """
    Tokenizes input text using BERT tokenizer and filters stopwords.

    Args:
        text (str): Input text.

    Returns:
        list: Clean list of tokens.
    """
    tokens = tokenizer.tokenize(text.lower())
    return [t for t in tokens if t not in STOPWORDS and re.match(r'[a-z]+', t)]

def tfidf_key(messages, top_n=5):
    """
    Extracts top N TF-IDF keywords using BERT-tokenized messages.

    Args:
        messages (list): List of all messages (user + AI).
        top_n (int): Number of keywords to return.

    Returns:
        list: Top N keywords sorted by TF-IDF score.
    """

    processed = [" ".join(bert_tokenize(msg)) for msg in messages]
    vectonizer = TfidfVectorizer()
    X = vectonizer.fit_transform(processed)
    scores = X.sum(axis=0).A1
    vocab = vectonizer.get_feature_names_out()
    keyword_scores = list(zip(vocab, scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    return [word for word, _ in keyword_scores[:top_n]]

def generate_summary(stats, keywords):
    """
    Creates a text summary of the chat statistics and top keywords.

    Args:
        stats (dict): Message statistics.
        keywords (list): Top keywords.

    Returns:
        str: Text summary.
    """

    return (
        f"\nSummary:\n"
        f"- Total exchanges: {stats['Total']}\n"
        f"- User messages: {stats['User']}\n"
        f"- AI messages: {stats['AI']}\n"
        f"- Top TF-IDF keywords: {', '.join(keywords)}\n"
    )

def plot_msg_stats(stats, filename=None):
    """
    Plots a bar chart of message counts using Seaborn and Matplotlib.

    Args:
        stats (dict): Message statistics.
        filename (str, optional): If provided, saves the plot to this file.
    """
    
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
