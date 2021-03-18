# ------------------------------------------------------
# CDS Web Scraping Workshop - 2/29/20
# by Tushar Khan
# ------------------------------------------------------

import requests, pandas as pd
from bs4 import BeautifulSoup

# Create dataframe
features = ['post score', 'post link', 'score', 'commenter', 'comment']
data = pd.DataFrame(columns=features)

# Request top posts page
base_url   = 'https://old.reddit.com/r/all/top/'
user_agent = {'User-Agent': 'Mozilla/5.0'}

r = requests.get(base_url, params={'t': 'all'}, headers=user_agent)

# Extract comment section links from the page
soup = BeautifulSoup(r.text, 'lxml')
link_tags = soup.find_all('a', class_='bylink comments may-blank')

links = [tag['href'] for tag in link_tags]

# Iterate over the links
for i, link in enumerate(links):
    print(f'Scraping data from link {i+1} of {len(links)}')

    r = requests.get(link, params={'sort': 'top'}, headers=user_agent)
    soup = BeautifulSoup(r.text, 'lxml')

    # Extract original post score
    score_str = soup.find('div', class_='linkinfo').find('span', class_='number').text
    op_score = int(score_str.replace(',', '')) # because `find` returns a string

    # Extract the comments from the comment section
    comment_filter = lambda t: t.find('span', class_='score unvoted') is not None
    comments = soup.find_all(comment_filter, attrs={'class': 'entry unvoted'})

    # Iterate over the comments
    for container in comments:
        comment = container.find('div', class_='usertext-body').text.strip().replace('\n', ' ')
        commenter = container.find(class_='author').text
        score = int(container.find(class_='score unvoted')['title'])

        # Append row to dataframe
        data = data.append(dict(zip(features, [op_score, link, score, commenter, comment])), ignore_index=True)

# Write dataframe to csv
data.to_csv('reddit_comments.csv', index=False)

# Inspect dataset
data.sort_values(by='score', ascending=False).head()