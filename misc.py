import pandas as pd
from global_vars import BLOGS_PATH


df = pd.read_csv(BLOGS_PATH)
blogs = df['text'].tolist()
labels = df['label'].tolist()

word_counts = [len(blog.split(" ")) for blog in blogs]
word_counts.sort()

total_blogs = len(blogs)

percent = round(total_blogs * 0.85)

print(word_counts[percent])