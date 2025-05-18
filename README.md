# 🧠 AI Chat Log Summarizer (BERT + TF-IDF)

This Python-based tool reads `.txt` chat logs between a **User** and an **AI**, parses the messages, and generates a concise summary including:

- ✅ Total message count
- ✅ User vs AI message breakdown
- ✅ Top keywords using **BERT tokenizer** and **TF-IDF scoring**
- ✅ Message count bar chart using **Seaborn/Matplotlib**

---

## 📁 Sample Chat Format (`chat_logs/chat.txt`)


```bash
User: Hello!
AI: Hi! How can I assist you today?
User: Can you explain what machine learning is?
AI: Certainly! Machine learning is a field of AI that allows systems to learn from data.


---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-chat-log-summarizer.git
cd ai-chat-log-summarizer

### 2. Install Dependencies

```bash
pip install -r requirements.txt

### 3. Add chat log files

Place your .txt chat logs in the chat_logs/ directory. The format must begin each message with either User: or AI:.

### 4. Run the summarizer

```bash
python main.py

## Example Output

```bash
Processing: chat.txt

Summary:
- Total exchanges: 4
- User messages: 2
- AI messages: 2
- Top TF-IDF keywords: machine, learning, data, systems, ai

## 💡 Technologies Used
- Python 3.7+
- Transformers (for BERT tokenizer)
- Scikit-learn (TF-IDF keyword extraction)
- NLTK (stopword removal)
- Matplotlib & Seaborn (for bar chart)


