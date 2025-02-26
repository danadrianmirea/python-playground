import requests

# GitHub username and personal access token
username = 'danadrianmirea'
token = ''

# Function to get the followers list with pagination
def get_followers(username, token):
    followers = []
    url = f'https://api.github.com/users/{username}/followers'
    
    while url:
        response = requests.get(url, auth=(username, token))
        if response.status_code == 200:
            followers.extend([follower['login'] for follower in response.json()])
            # Check if there's a next page
            if 'Link' in response.headers:
                links = response.headers['Link']
                next_link = None
                for link in links.split(','):
                    if 'rel="next"' in link:
                        # Extract the URL from the link and remove the angle brackets
                        next_link = link.split(';')[0].strip()[1:-1]
                url = next_link
            else:
                break
        else:
            print(f"Error fetching followers: {response.status_code}")
            break
    return followers

# Function to get the following list with pagination
def get_following(username, token):
    following = []
    url = f'https://api.github.com/users/{username}/following'
    
    while url:
        response = requests.get(url, auth=(username, token))
        if response.status_code == 200:
            following.extend([follower['login'] for follower in response.json()])
            # Check if there's a next page
            if 'Link' in response.headers:
                links = response.headers['Link']
                next_link = None
                for link in links.split(','):
                    if 'rel="next"' in link:
                        # Extract the URL from the link and remove the angle brackets
                        next_link = link.split(';')[0].strip()[1:-1]
                url = next_link
            else:
                break
        else:
            print(f"Error fetching following: {response.status_code}")
            break
    return following

# Function to unfollow a user
def unfollow_user(username, token, user_to_unfollow):
    url = f'https://api.github.com/user/following/{user_to_unfollow}'
    response = requests.delete(url, auth=(username, token))
    if response.status_code == 204:
        print(f"Successfully unfollowed {user_to_unfollow}")
    else:
        print(f"Error unfollowing {user_to_unfollow}: {response.status_code}")

# Function to find users who follow you back and unfollow those who don't
def unfollow_non_followers(username, token):
    followers = get_followers(username, token)
    following = get_following(username, token)
    
    non_followers = [user for user in following if user not in followers]
    
    if non_followers:
        print("Unfollowing people who do not follow you back:")
        for user in non_followers:
            unfollow_user(username, token, user)
    else:
        print("Everyone you follow follows you back!")

# Unfollow non-followers
unfollow_non_followers(username, token)
