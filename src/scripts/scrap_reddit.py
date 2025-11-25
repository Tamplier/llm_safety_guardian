import requests
import time
import pandas as pd
from src.util import PathHelper

headers = {'User-Agent': 'Mozilla/5.0'}

def get_posts(subreddit, max_posts, after=None):
    url = f"https://www.reddit.com/r/{subreddit}/hot.json"
    posts = []

    while len(posts) < max_posts:
        params = {'limit': 100, 'after': after}
        r = requests.get(url, headers=headers, params=params, timeout=15)

        if r.status_code != 200:
            time.sleep(60*5)
            continue

        data = r.json()['data']
        children = data['children']

        if not children:
            break

        for c in children:
            posts.append(c['data']['id'])

        after = data['after']
        if not after:
            break

        time.sleep(5)

    return after, posts[:max_posts]

def extract_comments(node, out):
    if node is None:
        return

    kind = node.get("kind")
    if kind != "t1":
        return

    body = node["data"].get("body")
    if body and body not in ["[deleted]", "[removed]"] and len(body) >= 15:
        out.append(body)

    replies = node["data"].get("replies")
    if isinstance(replies, dict):
        children = replies["data"].get("children", [])
        for child in children:
            extract_comments(child, out)

def get_comments_for_post(post_id):
    url = f"https://www.reddit.com/comments/{post_id}.json"
    r = None

    success = False
    while not success:
        r = requests.get(url, headers=headers, timeout=15)
        success = r.status_code == 200
        if not success:
            time.sleep(60*5)

    try:
        comments_section = r.json()[1]['data']['children']
    except Exception:
        return []

    out = []
    for c in comments_section:
        extract_comments(c, out)

    return out

def get_n_comments(subreddit, target_count):
    all_comments = []
    after = None
    loop_condition = True
    while loop_condition:
        after, posts = get_posts(subreddit, 100, after)
        for post_id in posts:
            comments = get_comments_for_post(post_id)
            all_comments.extend(comments)
            time.sleep(5)
        print(f"{subreddit}: {len(all_comments)} collected so far...")
        loop_condition = after and len(all_comments) < target_count

    return all_comments[:target_count]

for sub in ['TrueAskReddit', 'AskReddit', 'CasualConversation']:
    data = []
    for comment in get_n_comments(sub, target_count=35_000):
        if comment in ['[deleted]', '[removed]'] or len(comment) < 10:
            continue
        data.append({
            'text': comment.strip(),
            'class': 'non-suicide'
        })
    df = pd.DataFrame(data)
    duplicated = df.duplicated()
    file_path = PathHelper.data.raw.get_path(f'{sub}_comments.csv')
    df[~duplicated].to_csv(file_path, index=False)
