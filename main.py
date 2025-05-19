import os
from utils import parse_chat, msg_stats, tfidf_key, generate_summary, plot_msg_stats

def main():
    """
    Main driver function for summarizing chat logs.

    - Scans the `chat_logs/` folder for .txt files.
    - Parses each chat log.
    - Computes message statistics.
    - Extracts top TF-IDF keywords using BERT tokenization.
    - Prints a textual summary.
    - Plots message count using seaborn/matplotlib.
    """
    
    folder_path = 'chat_logs'
    chat_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    if not chat_files:
        print("No chat log files found in 'chat_logs/' directory.")
        return

    for filename in chat_files:
        filepath = os.path.join(folder_path, filename)
        print(f"\nProcessing: {filename}")

        # Parse messages
        user_msgs, ai_msgs = parse_chat(filepath)

        # Skip if file has no valid messages
        if not user_msgs and not ai_msgs:
            print("No valid messages found. Skipping.")
            continue

        # Generate stats, keywords, and summary
        stats = msg_stats(user_msgs, ai_msgs)
        keywords = tfidf_key(user_msgs + ai_msgs)
        summary = generate_summary(stats, keywords)

        # Print summary
        print(summary)

        # Plot message bar chart
        plot_msg_stats(stats, filename=f"outputs/{filename}_bar_chart.png")

if __name__ == "__main__":
    main()