import transformers
import torch
import pandas as pd
import random
from global_vars import MODEL_ID, TOPICS, TARGET_TEXT_PATH, BLOGS_PATH, \
    SHAKESPEARE_EXAMPLE, SAMPLE_TEXT_PROMPT, TARGET_TEXT_PROMPT, REWRITE_TEXT_PROMPT, \
    TEXT_GEN_PATH

# LLM
pipeline = transformers.pipeline(
    "text-generation", model=MODEL_ID, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)


# generate target texts using the 10 topics
def generate_target_texts():
    print("\nGenerating target texts...")
    data = { "text": [] }
    for topic in TOPICS:
        output = pipeline(topic, max_new_tokens=128, temperature=0.1, pad_token_id=pipeline.tokenizer.eos_token_id)
        data['text'].append(output[0]['generated_text'])
    

    save_data(data, TARGET_TEXT_PATH)

    return data['text']


# prompts the LLM to perform style transfer and saves results
def generate_style_transfer():
    print("\nPerforming Style Transfer...")
    
    # load blog samples
    blogs_df = pd.read_csv(BLOGS_PATH)
    blogs = blogs_df['text'].tolist()
    labels = blogs_df['label'].tolist()

    # load target texts
    target_texts_df = pd.read_csv(TARGET_TEXT_PATH)
    target_texts = target_texts_df['text'].tolist()

    blogs_table = {}

    # generate map of blogs and bloggers
    for target_idx in range(len(labels)):
        l = labels[target_idx]
        b = blogs[target_idx]

        if l not in blogs_table:
            blogs_table[l] = [b]
        else:
            blogs_table[l].append(b)
    
    # dict containing the LLM outputs to be saved
    outputs = {
        "target_idx": [],
        "blogger_id": [],
        "generated_text": []
    }

    # for each target text and blogger pair, style transfer is performed and stored in outputs
    for target_idx in range(len(target_texts)):
        print("\nTarget text:", target_idx)
        target_text = target_texts[target_idx]

        for label in blogs_table.keys():
            prompt = build_prompt(target_text, blogs_table[label])
            text_gen = pipeline(prompt, max_new_tokens=256, temperature=0.1, pad_token_id=pipeline.tokenizer.eos_token_id)

            # extract relevant part of output
            output = text_gen[0]["generated_text"][len(prompt):]
            output = output[:output.find("}")]

            outputs["target_idx"].append(target_idx)
            outputs["blogger_id"].append(label)
            outputs["generated_text"].append(output)
    
    save_data(outputs, TEXT_GEN_PATH)


# builds the prompt to initiate text style transfer
def build_prompt(target_text, all_samples):
    prompt = SHAKESPEARE_EXAMPLE

    prompt_samples = []  # samples to use in prompt
    searched_samples = []  # samples already searched
    total_chars = 0  # total character count of all the prompt samples
    chars_target = 256 * 4  # target character count that we want to use in the prompt
    
    # randomly selects samples from the given blogger until the character target is reached
    while total_chars < chars_target:
        sample_index = random.randint(0, len(all_samples) - 1)
        if sample_index not in searched_samples:
            sample = all_samples[sample_index]
            sample_chars = len(sample)
            if total_chars + sample_chars < chars_target:
                prompt_samples.append(sample)
                total_chars += sample_chars
            searched_samples.append(sample_index)
    
    sample = " ".join(prompt_samples)  # prompt_samples concatenated

    prompt += SAMPLE_TEXT_PROMPT('B', sample) + TARGET_TEXT_PROMPT(target_text) + REWRITE_TEXT_PROMPT('B')

    return prompt


# save data to csv
def save_data(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

    print("\nSaved " + filename + "!")
