import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional
from torch.utils.data import Dataset


def benchmark(func):
    from datetime import datetime

    def wrapper(*args, **kwargs):
        t = datetime.now()
        res = func(*args, **kwargs)
        print(func.__name__, datetime.now() - t)
        return res

    return wrapper


@dataclass
class ModelConfig:
    block_size: int = None
    vocab_size: int = None
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4


class CharDataset(Dataset):

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1

    def get_output_length(self):
        return self.max_word_length + 1

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1 + len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix) + 1:] = -1
        return x, y


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        b, t, c = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :t, :t] == 0, float('-inf'))
        att = functional.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.c_proj(y)
        return y


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    @staticmethod
    def forward(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
            act=NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = functional.softmax(logits, dim=-1)
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_samples(num, model, train_dataset, test_dataset):
    """ samples from the model and pretty prints the decoded samples """
    x_init = torch.zeros(num, 1, dtype=torch.long).to('cpu')
    steps = train_dataset.get_output_length() - 1
    x_sample = generate(model, x_init, steps, top_k=None, do_sample=True).to('cpu')
    train_samples, test_samples, new_samples = [], [], []
    for i in range(x_sample.size(0)):
        row = x_sample[i, 1:].tolist()
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_sample = train_dataset.decode(row)
        if train_dataset.contains(word_sample):
            train_samples.append(word_sample)
        elif test_dataset.contains(word_sample):
            test_samples.append(word_sample)
        else:
            new_samples.append(word_sample)
    return {
        "in train": train_samples,
        "in test": test_samples,
        "new": new_samples
    }


def create_datasets(input_file):
    with open(input_file, 'r', encoding="utf-8") as f:
        data = f.read().lower()
    words = data.splitlines()
    words = [w.strip() for w in words]
    words = [w for w in words if w]
    chars = sorted(list(set(''.join(words))))
    max_word_length = max(len(w) for w in words)
    test_set_size = min(1000, int(len(words) * 0.1))
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)
    return train_dataset, test_dataset


def fix_multi_chars(words_list, max_char=2):
    for i in range(len(words_list)):
        stop = False
        while not stop:
            counter = 0
            fix = False
            cur_c = ""
            for c in words_list[i]:
                if cur_c != c:
                    cur_c = c
                    counter = 1
                else:
                    counter += 1
                if counter > max_char:
                    fix = True
                    break
            if fix:
                print(words_list[i])
                words_list[i] = words_list[i].replace(cur_c * counter, cur_c * max_char)
                print(words_list[i])
            else:
                stop = True


def delete_stumps(words_list):
    cleaned_list = []
    for i in words_list:
        if len(i) > 1:
            cleaned_list.append(i)
    return cleaned_list


@benchmark
def gen(requested_count, input_wordlist_path, model_path):
    train_dataset, test_dataset = create_datasets(input_wordlist_path)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size)
    model = Transformer(config)
    model.load_state_dict(torch.load(model_path, torch.device('cpu')))
    samples = []
    count = requested_count
    while len(samples) != requested_count:
        new_samples = generate_samples(count, model, train_dataset, test_dataset)["new"]
        cleaned_new_samples = delete_stumps(new_samples)
        fix_multi_chars(cleaned_new_samples)
        samples.extend(cleaned_new_samples)
        count = requested_count - len(samples)
    return samples


# use sample
if __name__ == '__main__':
    print(gen(
        requested_count=100,
        input_wordlist_path="ru_male_names.txt",
        model_path="models/ru_male_names.pt"
    ))
