import argparse
import math
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter


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


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    @staticmethod
    def forward(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


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
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y


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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


class CausalBoW(nn.Module):
    """
    Causal bag of words. Averages the preceding elements and looks suspiciously like
    a CausalAttention module you'd find in a transformer, for no apparent reason at all ;)
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        att = torch.zeros((B, T, T), device=x.device)
        att = att.masked_fill(self.bias[:, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ x
        return y


class BoWBlock(nn.Module):
    """ collects BoW features and adds an MLP """

    def __init__(self, config):
        super().__init__()
        self.cbow = CausalBoW(config)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.n_embd, config.n_embd2),
            c_proj=nn.Linear(config.n_embd2, config.n_embd),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(F.tanh(m.c_fc(x)))

    def forward(self, x):
        x = x + self.cbow(x)
        x = x + self.mlpf(x)
        return x


class BoW(nn.Module):
    """
    takes the previous block_size tokens, encodes them with a lookup table,
    also encodes their positions with lookup table, then averages all of those
    embeddings up and uses that to predict the next token.
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.context_block = BoWBlock(config)
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        x = self.context_block(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


"""
Recurrent Neural Net language model: either a vanilla RNN recurrence or a GRU.
Did not implement an LSTM because its API is a bit more annoying as it has
both a hidden state and a cell state, but it's very similar to GRU and in
practice works just as well.
"""


class RNNCell(nn.Module):
    """

    the job of a 'Cell' is to:
    take input at current time step x_{t} and the hidden state at the
    previous time step h_{t-1} and return the resulting hidden state
    h_{t} at the current timestep
    """

    def __init__(self, config):
        super().__init__()
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1)
        ht = F.tanh(self.xh_to_h(xh))
        return ht


class GRUCell(nn.Module):
    """
    same job as RNN cell, but a bit more complicated recurrence formula
    that makes the GRU more expressive and easier to optimize.
    """

    def __init__(self, config):
        super().__init__()
        self.xh_to_z = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_r = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_hbar = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1)
        r = F.sigmoid(self.xh_to_r(xh))
        hprev_reset = r * hprev

        xhr = torch.cat([xt, hprev_reset], dim=1)
        hbar = F.tanh(self.xh_to_hbar(xhr))

        z = F.sigmoid(self.xh_to_z(xh))

        ht = (1 - z) * hprev + z * hbar
        return ht


class RNN(nn.Module):

    def __init__(self, config, cell_type):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.start = nn.Parameter(torch.zeros(1, config.n_embd2))
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        if cell_type == 'rnn':
            self.cell = RNNCell(config)
        elif cell_type == 'gru':
            self.cell = GRUCell(config)
        self.lm_head = nn.Linear(config.n_embd2, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        b, t = idx.size()

        emb = self.wte(idx)

        hprev = self.start.expand((b, -1))
        hiddens = []
        for i in range(t):
            xt = emb[:, i, :]
            ht = self.cell(xt, hprev)
            hprev = ht
            hiddens.append(ht)

        hidden = torch.stack(hiddens, 1)
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


class MLP(nn.Module):
    """
    takes the previous block_size tokens, encodes them with a lookup table,
    concatenates the vectors and predicts the next token with an MLP.

    Reference:
    Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size)
        )

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):

        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size
            embs.append(tok_emb)

        x = torch.cat(embs, -1)
        logits = self.mlp(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


class Bigram(nn.Module):
    """
    Bigram Language Model 'neural net', simply a lookup table of logits for the
    next character given a previous character.
    """

    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))

    @staticmethod
    def get_block_size():
        return 1

    def forward(self, idx, targets=None):
        logits = self.logits[idx]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False):
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
        probs = F.softmax(logits, dim=-1)
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to('cpu') for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train()
    return mean_loss


class CharDataset(Dataset):

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def __len__(self):
        return len(self.words)

    def __contains__(self, item):
        return item in self.words

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


class InfiniteDataLoader:
    """
    this is really hacky, and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch


def create_dataset(input_file, silent=True):
    with open(input_file, 'r', encoding="utf-8") as f:
        data = f.read().lower()
    words = data.splitlines()
    words = [w.strip() for w in words]
    words = [w for w in words if w]
    chars = sorted(list(set(''.join(words))))
    max_word_length = max(len(w) for w in words)
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp]
    train_dataset = CharDataset(train_words, chars, max_word_length)
    if not silent:
        from rich import print
        print(f"> number of examples in the dataset: {len(words)}")
        print(f"> max word length: {max_word_length}")
        print(f"> number of unique characters in the vocabulary: {len(chars)}")
        print("> vocabulary: " + "".join(chars))
    return train_dataset


def generate_samples(num, model, train_dataset):
    """ new samples from the model and pretty prints the decoded samples """
    x_init = torch.zeros(num, 1, dtype=torch.long).to('cpu')
    steps = train_dataset.get_output_length() - 1
    x_sample = generate(model, x_init, steps, do_sample=True).to('cpu')
    new_samples = []
    for i in range(x_sample.size(0)):
        row = x_sample[i, 1:].tolist()
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_sample = train_dataset.decode(row)
        if word_sample not in train_dataset:
            new_samples.append(word_sample)
    return new_samples


def get_model(config, model_type):
    if model_type == 'transformer':
        return Transformer(config)
    elif model_type == 'bigram':
        return Bigram(config)
    elif model_type == 'mlp':
        return MLP(config)
    elif model_type == 'rnn':
        return RNN(config, cell_type='rnn')
    elif model_type == 'gru':
        return RNN(config, cell_type='gru')
    elif model_type == 'bow':
        return BoW(config)
    else:
        raise ValueError(f'model type {model_type} is not recognized')


@benchmark
def gen(requested_count: int, input_wordlist_path: str, model_path: str, silent: bool = True,
        n_layer: int = 8, n_head: int = 8, n_embd: int = 128, n_embd2: int = 128, model_type='transformer'):
    def delete_stumps(words_list):
        cleaned_list = []
        for i in words_list:
            if len(i) > 1:
                cleaned_list.append(i)
        return cleaned_list

    train_dataset = create_dataset(input_wordlist_path)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                         n_layer=n_layer, n_head=n_head,
                         n_embd=n_embd, n_embd2=n_embd2)
    model = get_model(config, model_type)
    model.load_state_dict(torch.load(model_path, torch.device('cpu')))
    samples = []

    from rich.progress import Progress
    with Progress() as progress:
        if not silent:
            task = progress.add_task("[red]Generating...", total=requested_count)
        while len(samples) != requested_count:
            count = 100 if requested_count > 100 else requested_count - len(samples)
            new_samples = generate_samples(count, model, train_dataset)
            cleaned_new_samples = delete_stumps(new_samples)
            for s in cleaned_new_samples:
                if s not in samples:
                    samples.append(s)
                    if not silent:
                        progress.update(task, advance=1)
    return samples


def ml_terminal():
    from datetime import datetime, timedelta
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress
    from rich import print

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-i', type=str, default='names.txt', help="input file with things one per line")
    parser.add_argument('--model-name', '-o', type=str, default='default', help="output model")
    parser.add_argument('--work-dir', '-d', type=str, default='models', help="output working directory")
    parser.add_argument('--sample-dir', type=str, default='samples', help="output samples directory")
    parser.add_argument('--resume', action='store_true',
                        help="when this flag is used, we will resume optimization from existing model in the workdir")
    parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
    parser.add_argument('--num-workers', '-n', type=int, default=4, help="number of data workers for both train/test")
    parser.add_argument('--max-steps', '-m', type=int, default=-1,
                        help="max number of optimization steps to run for, or -1 for infinite.")
    parser.add_argument('--seed', type=int, default=1998, help="seed")
    parser.add_argument('--type', type=str, default='transformer',
                        help="model class type to use, bigram|mlp|rnn|gru|bow|transformer")
    parser.add_argument('--n-layer', type=int, default=4, help="number of layers")
    parser.add_argument('--n-head', type=int, default=4, help="number of heads (in a transformer)")
    parser.add_argument('--n-embd', type=int, default=64, help="number of feature channels in the model")
    parser.add_argument('--n-embd2', type=int, default=64, help="number of feature channels elsewhere in the model")
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay")
    parser.add_argument('--time-limit', '-t', type=int, default=-1, help="time limit in minutes")
    parser.add_argument('--auto-save', '-a', type=int, default=500,
                        help="once in how many epochs does auto-save occur")

    args = parser.parse_args()
    print(vars(args))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=args.work_dir)

    train_dataset = create_dataset(args.input_file, silent=False)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"> dataset determined that: {vocab_size=}, {block_size=}")

    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                         n_layer=args.n_layer, n_head=args.n_head,
                         n_embd=args.n_embd, n_embd2=args.n_embd2)

    model = get_model(config, args.type)
    model.to('cpu')
    print(f"> model parameters: {sum(p.numel() for p in model.parameters())}")

    model_name: str = args.model_name + ".pt"
    if args.resume or args.sample_only:
        print("> resuming from existing model in the workdir")
        model.load_state_dict(torch.load(os.path.join(args.work_dir, model_name),
                                         map_location=torch.device('cpu')))
        os.makedirs(args.sample_dir, exist_ok=True)

    console = Console()
    status = console.status("[bold green]Preparing to work...")
    progress = Progress(transient=True)
    panel = Panel("Wait to examples...", title="Examples", border_style="blue", style="blue")

    if args.sample_only:
        samples = generate_samples(50, model, train_dataset)
        panel.renderable = "\n".join(samples)
        with open(os.path.join(args.sample_dir, args.model_name + ".txt"), 'w', encoding="utf-8") as f:
            f.write("\n".join(samples))
        print(panel)
        sys.exit()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.99), eps=1e-8)

    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True,
                                      num_workers=args.num_workers)

    best_loss = None

    ml_panel = Panel(Group(status, progress, panel), title="ML Process")
    time_limit = timedelta(minutes=args.time_limit)

    start_time = datetime.now()
    print(f"Start time {start_time}")
    with (Live(ml_panel)):

        if args.max_steps != -1:
            task_id = progress.add_task("[bold green]Progress")
            tracker = progress.track(range(args.max_steps), task_id=task_id)
        else:
            tracker = range(2 ** 1024)

        for step in tracker:

            t0 = time.time()

            batch = batch_loader.next()
            batch = [t.to('cpu') for t in batch]
            X, Y = batch

            logits, loss = model(X, Y)

            model.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                t1 = time.time()
                status.update(f"[bold green]Learning..."
                              f" Step {step} | loss {loss.item():.4f} | step time {(t1 - t0) * 1000:.2f}ms")

            if step > 0 and step % args.auto_save == 0:
                train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
                writer.add_scalar("Loss/train", train_loss, step)
                writer.flush()

                if best_loss is None or train_loss < best_loss:
                    out_path = os.path.join(args.work_dir, model_name)
                    torch.save(model.state_dict(), out_path)
                    best_loss = train_loss
                    ml_panel.title = f"ML Process (step save:{step}, best loss: {best_loss})"
                    samples = generate_samples(10, model, train_dataset)
                    panel.renderable = "\n".join(samples)
                    console.render(panel)

            if (args.max_steps != -1 and args.max_steps <= step) \
                    or (args.time_limit != -1 and datetime.now() - start_time > time_limit):
                del tracker
                break
    end_time = datetime.now()
    print(f"End time {end_time}. Time delta {end_time - start_time}.")
    sys.exit()


if __name__ == '__main__':
    ml_terminal()
