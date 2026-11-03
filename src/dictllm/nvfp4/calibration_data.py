"""Unchanged research tokenization/grouping helpers, without legacy model imports."""

def tokenize_func(example, tokenizer, content):
    return tokenizer(example[content])

def group_text(examples, context_length):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // context_length) * context_length
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + context_length] for i in range(0, total_length, context_length)]
        for k, t in concatenated_examples.items()
    }
    return result
