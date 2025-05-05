from torch import nn
from transformers import BertModel

# data file paths
CORPUS_PATH = "data/corpus.csv"
BLOGS_PATH = "data/blogs.csv"
BLOGS_TRAIN_PATH = "data/blogs_train.csv"
BLOGS_TEST_PATH = "data/blogs_test.csv"
TARGET_TEXT_PATH = "data/target_texts.csv"

# bert file paths
BERT_PATH = "bert/bert_classifier.pth"
TRAINING_IMG_PATH = "bert/validation_accuracy.png"

# results file paths
TEXT_GEN_PATH = "results/text_gen.csv"

# topics
TOPICS = [
    'I love cars',
    'What do you know about fashion?',
    'Television is so boring nowadays',
    'Gaming is dead',
    'Reading books',
    'Art exhibition',
    'Gardening is fun',
    'Travelling the world',
    'Family is everything',
    'Football highlights'
]

# prompting
SAMPLE_TEXT_PROMPT = lambda author, sample: "Here is a sample of some text in Author "+author+"'s writing style: {"+sample+"}. "
TARGET_TEXT_PROMPT = lambda text: "Here is the target text: {"+text+"}. "
REWRITE_TEXT_PROMPT = lambda author: "Here is the target text rewritten in Author "+author+"'s writing style: {"
SHAKESPEARE_EXAMPLE = SAMPLE_TEXT_PROMPT('A', "When shall we three meet again? In thunder, lightning, or in rain? Fair is foul, and foul is fair. Hover through the fog and filthy air.")+TARGET_TEXT_PROMPT("We'll meet when the noise of the battle is over, when one side has won and the other side has lost.")+REWRITE_TEXT_PROMPT('A')+"When the hurly-burly's done, When the battle's lost and won.}. "

# LLM model ID
MODEL_ID = "meta-llama/Llama-3.2-3B-instruct"