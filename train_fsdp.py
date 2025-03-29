# %% [markdown]
# A cute little demo showing the simplest usage of minGPT. Configured to run fine on Macbook Air in like a minute.

# %%
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import set_seed
set_seed(3407)

# %%
import pickle

class SortDataset(Dataset):
    """ 
    Dataset for the Sort problem. E.g. for problem length 6:
    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
    Which will feed into the transformer concatenated as:
    input:  0 0 2 1 0 1 0 0 0 1 1
    output: I I I I I 0 0 0 1 1 2
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(self, split, length=6, num_digits=3):
        assert split in {'train', 'test'}
        self.split = split
        self.length = length
        self.num_digits = num_digits
    
    def __len__(self):
        return 10000 # ...
    
    def get_vocab_size(self):
        return self.num_digits
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length * 2 - 1

    def __getitem__(self, idx):
        
        # use rejection sampling to generate an input example from the desired split
        while True:
            # generate some random integers
            inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
            # half of the time let's try to boost the number of examples that 
            # have a large number of repeats, as this is what the model seems to struggle
            # with later in training, and they are kind of rate
            if torch.rand(1).item() < 0.5:
                if inp.unique().nelement() > self.length // 2:
                    # too many unqiue digits, re-sample
                    continue
            # figure out if this generated example is train or test based on its hash
            h = hash(pickle.dumps(inp.tolist()))
            inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
            if inp_split == self.split:
                break # ok
        
        # solve the task: i.e. sort
        sol = torch.sort(inp)[0]

        # concatenate the problem specification and the solution
        cat = torch.cat((inp, sol), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input locations
        y[:self.length-1] = -1
        return x, y


# %%
# print an example instance of the dataset
train_dataset = SortDataset('train')
test_dataset = SortDataset('test')
x, y = train_dataset[0]
for a, b in zip(x,y):
    print(int(a),int(b))

# %%
import json
with open('configs/benchmark.json') as f:
    config = json.load(f)

# %%
# create a GPT instance
from mingpt.model import GPT

model_config = GPT.get_default_config()
model_config.model_type = config['model_config']['model_type']
model_config.vocab_size = train_dataset.get_vocab_size()
model_config.block_size = train_dataset.get_block_size()
model_config.use_checkpoint = config['model_config']['use_checkpoint']
model = GPT(model_config)

# %%
# create a Trainer object
from mingpt.trainer import Trainer

train_config = Trainer.get_default_config()
train_config.learning_rate = config['train_config']['learning_rate']
train_config.batch_size = config['train_config']['batch_size']
train_config.cross_batch_num = config['train_config']['cross_batch_num']
train_config.num_workers = config['train_config']['num_workers']
train_config.total_num_train_data = config['train_config']['total_num_train_data']
train_config.max_iters = int(train_config.total_num_train_data / train_config.batch_size/train_config.cross_batch_num)
train_config.device = config['train_config']['device']
train_config.foreach_opt_flags = config['train_config']['foreach_opt_flags']
train_config.use_fsdp = config['train_config']['use_fsdp']
trainer = Trainer(train_config, model, train_dataset)

# %%
import subprocess
import time
import threading
import matplotlib.pyplot as plt

# Function to monitor GPU memory and save data to a list
def monitor_gpu_memory(gpu_index=1, interval=5, duration=60):
    memory_usage = []
    start_time = time.time()
    while time.time() - start_time < duration:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader', '-i', str(gpu_index)], stdout=subprocess.PIPE)
        memory_used = int(result.stdout.decode('utf-8').strip())
        memory_usage.append(memory_used)
        print(f"GPU {gpu_index} Memory Used: {memory_used} MiB")
        time.sleep(interval)
    return memory_usage

# Function to plot memory usage data
def plot_memory_usage(memory_usage):
    print(f"Max memory usage: {max(memory_usage)} - Min memory usage: {memory_usage[0]}")
    plt.plot(memory_usage)
    plt.xlabel('Time (intervals)')
    plt.ylabel('Memory Used (MiB)')
    plt.title('GPU Memory Usage Over Time')
    plt.show()




# %%
# Start GPU memory monitoring in a separate thread
monitor_thread_0 = threading.Thread(target=lambda: plot_memory_usage(monitor_gpu_memory(gpu_index=2, interval=0.001, duration=10)))
monitor_thread_0.start()

monitor_thread_1 = threading.Thread(target=lambda: plot_memory_usage(monitor_gpu_memory(gpu_index=3, interval=0.001, duration=10)))
monitor_thread_1.start()

# %%
time_list = []
def batch_end_callback(trainer):
    if trainer.iter_num % 10 == 0 and trainer.iter_num > 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        time_list.append(trainer.iter_dt * 1000)
trainer.set_callback('on_batch_end', batch_end_callback)

trainer.run()
print("average time per iteration: ", sum(time_list) / len(time_list) * train_config.max_iters, "ms")

# %%
monitor_thread_0.join()
monitor_thread_1.join()


# %%
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('run/model_structure')
# batch = [t.to(tranier.device) for t in next(iter(trainer.train_loader))]
# writer.add_graph(model, batch)
# writer.close

# %%
# !tensorboard --logdir runs

# %%
# now let's perform some evaluation
model.eval();

# %%
def eval_split(trainer, split, max_batches):
    dataset = {'train':train_dataset, 'test':test_dataset}[split]
    n = train_dataset.length # naugy direct access shrug
    results = []
    mistakes_printed_already = 0
    loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
    for b, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        y = y.to(trainer.device)
        # isolate the input pattern alone
        inp = x[:, :n]
        sol = y[:, -n:]
        # let the model sample the rest of the sequence
        cat = model.generate(inp, n, do_sample=False) # using greedy argmax, not sampling
        sol_candidate = cat[:, n:] # isolate the filled in sequence
        # compare the predicted sequence to the true sequence
        correct = (sol == sol_candidate).all(1).cpu() # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            if not correct[i] and mistakes_printed_already < 3: # only print up to 5 mistakes to get a sense
                mistakes_printed_already += 1
                print("GPT claims that %s sorted is %s but gt is %s" % (inp[i].tolist(), sol_candidate[i].tolist(), sol[i].tolist()))
        if max_batches is not None and b+1 >= max_batches:
            break
    rt = torch.tensor(results, dtype=torch.float)
    print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
    return rt.sum()

# run a lot of examples from both train and test through the model and verify the output correctness
with torch.no_grad():
    train_score = eval_split(trainer, 'train', max_batches=50)
    test_score  = eval_split(trainer, 'test',  max_batches=50)

# %%
# let's run a random given sequence through the model as well
n = train_dataset.length # naugy direct access shrug
inp = torch.tensor([[0, 0, 2, 1, 0, 1]], dtype=torch.long).to(trainer.device)
assert inp[0].nelement() == n
with torch.no_grad():
    cat = model.generate(inp, n, do_sample=False)
sol = torch.sort(inp[0])[0]
sol_candidate = cat[:, n:]
print('input sequence  :', inp.tolist())
print('predicted sorted:', sol_candidate.tolist())
print('gt sort         :', sol.tolist())
print('matches         :', bool((sol == sol_candidate).all()))


