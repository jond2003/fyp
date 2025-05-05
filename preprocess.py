import pandas as pd
from global_vars import CORPUS_PATH, BLOGS_PATH

# narrow down dataset to 20 bloggers and preprocess data
def reduce_blogs(data_file):
    print("\nPreprocessing blogs...")
    # read dataset and get text and user ids
    df = pd.read_csv(data_file)
    blogs = df['text'].tolist()
    ids = df['id'].tolist()

    labels_table = {}  # maps user id to training labels
    blogger_wordcount = {}  # total word count for each blogger
    normalised_labels = []  # training label for each blog after preprocessing
    normalised_blogs = []  # preprocessed blogs

    bloggers_blacklist = [3403444, 2822042]  # blogger ids to ignore

    # gets total word count for each blogger
    for label in range(len(blogs)):
        id = ids[label]
        blog = blogs[label]
        if id not in blogger_wordcount:
            blogger_wordcount[id] = len(blog.split(" "))
        else:
            blogger_wordcount[id] += len(blog.split(" "))

    label = 0
    print("ID : Label")
    for i in range(len(ids)):
        id = ids[i]
        # 200k = 41 bloggers
        # 250k = 22 bloggers
        if blogger_wordcount[id] > 250000 and id not in bloggers_blacklist:
            if id not in labels_table:
                labels_table[id] = label
                print(id, ": ", label)
                label += 1
            
            # Only accept blogs greater than 16 words in length
            if len(blogs[i].strip().split(" ")) > 16:
                normalised_labels.append(labels_table[id])
                normalised_blogs.append(blogs[i].strip())
    
    print("\nSuccessfully preprocessed blogs.")
    return normalised_blogs, normalised_labels


# save blog data to csv
def save_data(blogs, labels, filename):
    data = {
        "label": labels,
        "text": blogs
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

    print("\nSaved " + filename + "!")

def preprocess_blogs():
    blogs, labels = reduce_blogs(CORPUS_PATH)
    save_data(blogs, labels, BLOGS_PATH)  # narrowed blogs