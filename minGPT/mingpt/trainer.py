"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
import torch.distributed as dist
import multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload
import os
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from mingpt.utils import CfgNode as CN

class TorchProfiler:
    def __init__(
        self,
        log_dir="./log",
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
        wait=1,
        warmup=1,
        active=3,
        repeat=2,
    ):
        self.log_dir = log_dir
        self.profile_memory = profile_memory
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.schedule = schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)

    def __enter__(self):
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=self.schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
            profile_memory=self.profile_memory,
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
        )
        self.profiler.__enter__()
        return self.profiler

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.__exit__(exc_type, exc_val, exc_tb)
        print(self.profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))


class Trainer:
    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.cross_batch_num = 1
        C.use_fsdp = False
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.use_fsdp = confit.use_fsdp

        # determine the device we'll train on
        if self.use_fsdp:
            dist.init_process_group(backend="nccl")
            self.local_rank = int(os.env.get("LOCAL_RANK",0))
            print(">>> local_rank: ", self.local_rank)
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            self.model = self.model.to(self.local_rank)
            self.model = FSDP(self.model)
        else:
            if config.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = config.device
            self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        if self.use_fsdp:
            train_sampler = DistributedSampler(self.train_dataset)
            self.train_loader = DataLoader(
                self.train_dataset,
                sampler=train_sampler,
                shuffle=False,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )
        else:
            train_sampler = torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10))
            self.train_loader = DataLoader(
                self.train_dataset,
                sampler=train_sampler,
                shuffle=False,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(self.train_loader)
        steps = self.config.cross_batch_num
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            for i in range(steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)
                batch = [t.to(self.device) for t in batch]
                x, y = batch

                # forward the model
                logits, self.loss = model(x, y)

                # backprop and update the parameters
                model.zero_grad(set_to_none=True)
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

        if self.use_fsdp:
            dist.destroy_process_group()
