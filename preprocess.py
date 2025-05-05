import pandas as pd
from global_vars import CORPUS_PATH, BLOGS_PATH, BLOGS_TRAIN_PATH, BLOGS_TEST_PATH

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


# split blogs 50/50 into training and testing dataset
def split_data(blogs, labels):
    print("\nSplitting data...")
    train_blogs = []
    train_labels = []
    test_blogs = []
    test_labels = []

    unique_labels = set(labels)
    total_blogs_count = 0  # counter for blog index

    for label in unique_labels:
        blogs_count = labels.count(label)
        midpoint = total_blogs_count + round(blogs_count / 2)

        # first half of blogs assigned to training data
        train_blogs.extend(blogs[total_blogs_count:midpoint])
        train_labels.extend(labels[total_blogs_count:midpoint])

        total_blogs_count += blogs_count

        # second half of blogs assigned to test data
        test_blogs.extend(blogs[midpoint:total_blogs_count])
        test_labels.extend(labels[midpoint:total_blogs_count])
    
    print("\nSuccessfully split data into training and test data.")

    return train_blogs, train_labels, test_blogs, test_labels


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
    train_blogs, train_labels, test_blogs, test_labels = split_data(blogs, labels)

    save_data(blogs, labels, BLOGS_PATH)  # narrowed blogs
    save_data(train_blogs, train_labels, BLOGS_TRAIN_PATH)  # training data
    save_data(test_blogs, test_labels, BLOGS_TEST_PATH)  # testing data