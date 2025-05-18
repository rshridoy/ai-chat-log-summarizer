import os
from utils import parse_chat, msg_stats, tfidf_key, generate_summary, plot_msg_stats

def main():
    folder_path = 'chat_logs'
    chat_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    for filename in chat_files:
        filepath = os.path.join(folder_path, filename)
        print(f"\nProcessing: {filename}")

        user_msgs, ai_msgs = parse_chat(filepath)
        stats = msg_stats(user_msgs, ai_msgs)
        keywords = tfidf_key(user_msgs + ai_msgs)
        summary = generate_summary(stats, keywords)

        print(summary)
        plot_msg_stats(stats)

if __name__ == "__main__":
    main()