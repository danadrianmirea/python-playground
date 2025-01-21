import praw
import os
import time

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id="",
    client_secret="",
    user_agent="subScraper by /u/Positive_Lunch",
)

def search_subreddit(subreddit_name, query, filename, limit=100, commentLimit=10, logTime=5):
    """
    Searches for posts containing the specified query string in a subreddit and saves them into a text file.
    Displays progress every `logTime` seconds.

    :param subreddit_name: Name of the subreddit to search.
    :param query: The string to search for in the post title and selftext.
    :param filename: File to save the parsed posts and comments.
    :param limit: Maximum number of posts to fetch.
    :param commentLimit: Limit for comments to fetch.
    :param logTime: Time interval in seconds to display progress.
    """
    subreddit = reddit.subreddit(subreddit_name)
    
    start_time = time.time()
    post_count = 0

    # Fetch the posts using search and sort them by creation date
    posts = []
    for post in subreddit.search(query, limit=limit):
        posts.append(post)
    
    # Sort posts by creation date (newest first)
    posts.sort(key=lambda x: x.created_utc, reverse=True)
    
    with open(filename, "w", encoding="utf-8") as file:
        file.write("*" * 80 + "\n")
        
        for post in posts:
            post_count += 1
            file.write(f"Title: {post.title} ({post.url})\n")
            cleaned_post = post.selftext.replace("\n", " ")
            file.write(f"{cleaned_post}\n")
            
            # Get the comments
            if commentLimit != -1:
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


numPosts=1000
numCommentsPerPost=10
search_query = "c++"

if __name__ == "__main__":
    subreddit_name = "programare"
    output_file = subreddit_name +".txt"
    search_subreddit(subreddit_name, search_query, output_file, numPosts, numCommentsPerPost)
