# data/wikitext/prepare.py
import os
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")

def encode_file(input_path, output_path):
    with open(os.path.join(os.path.dirname(__file__), input_path), "r", encoding="utf-8") as f:
        text = f.read()
    ids = enc.encode_ordinary(text)
    arr = np.array(ids, dtype=np.int32) 
    arr.tofile(os.path.join(os.path.dirname(__file__), output_path))
    print(f"{input_path} -> {output_path}, tokens: {len(ids)}")

encode_file("wikitext_train.txt", "train.bin")
encode_file("wikitext_val.txt", "val.bin")

meta = {
    "vocab_size": 50257,
    "itos": None, 
    "stoi": None,
}
import pickle
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)