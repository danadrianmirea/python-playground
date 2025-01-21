import praw
import os
import time

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id="5upF4oYw0y3itVvyyC2aOQ",
    client_secret="SMLNeZaB0DCiPLw2j96qzmubLK3mYQ",
    user_agent="subScraper by /u/Positive_Lunch",
)

def parse_subreddit(subreddit_name, filename, limit=100, commentLimit=10, logTime=5):
    """
    Parses posts and their comments from a subreddit and saves them into a text file.
    Displays progress every `logTime` seconds.

    :param subreddit_name: Name of the subreddit to parse.
    :param filename: File to save the parsed posts and comments.
    :param limit: Maximum number of posts to fetch.
    :param logTime: Time interval in seconds to display progress.
    """
    subreddit = reddit.subreddit(subreddit_name)
    
    start_time = time.time()
    post_count = 0

            
    with open(filename, "w", encoding="utf-8") as file:
        file.write("*" * 80 + "\n")
        for post in subreddit.new(limit=limit):
            post_count += 1
            file.write(f"Title: {post.title} ({post.url})\n")
            cleaned_post = post.selftext.replace("\n", " ")
            file.write(f"{cleaned_post}\n")
            
            #file.write(f"Title: {post.title} ({post.url})\n")
            #file.write(f"Author: {post.author}\n")
            #file.write(f"Score: {post.score}\n")
            #file.write(f"Created: {post.created_utc}\n")
            #file.write(f"Content: {cleaned_post}\n")
            #file.write(f"URL: {post.url}\n")
            
            # Get the comments
            if numCommentsPerPost != -1:
                post.comments.replace_more(limit=commentLimit)  # Replace "More Comments" with actual comments
                sorted_comments = sorted(post.comments.list(), key=lambda x: x.score, reverse=True)
                for comment in sorted_comments:
                    cleaned_comment = comment.body.replace("\n", " ")
                    file.write(f"{comment.author}, {comment.score} votes: {cleaned_comment}\n")
            
            file.write("*" * 80 + "\n")

            # Check the elapsed time and print progress
            elapsed_time = time.time() - start_time
            if elapsed_time >= logTime:
                print(f"Parsed {post_count}/{limit} posts in {int(elapsed_time)} seconds.")

    print(f"Finished parsing {post_count} posts from r/{subreddit_name} into {filename}.")

numPosts=100
numCommentsPerPost=10

if __name__ == "__main__":
    subreddit_name = "programare"
    output_file = subreddit_name +".txt"
    parse_subreddit(subreddit_name, output_file, numPosts, numCommentsPerPost)
