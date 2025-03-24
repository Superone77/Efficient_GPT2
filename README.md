# Efficient GPT2

## Core Idea  
Trade off between model accuracy, training performance, and GPU memory consumption.

---

## Theoretical Foundations

### Benchmark Testing  
Monitor and print the GPU memory usage during the model training process to evaluate memory efficiency.

### Cross-Batch Gradient Accumulation  
By accumulating gradients over multiple training iterations and updating the weights only after several mini-batches, you effectively simulate a larger batch size. This approach can significantly lower the peak memory allocation since gradients from several small batches are combined before the update. However, this comes at the cost of training speed compared to processing a full batch in each iteration.

### On-the-Fly Recomputing of Forward Tensors  
Using tools like `torch.utils.checkpoint`, you can avoid storing intermediate forward tensors during the forward pass. Instead, these tensors are recomputed during the backward pass. This trade-off increases computation time but significantly reduces the memory needed to store activations, which is especially useful for very deep models.

### Offloading GPU Memory to CPU  
This technique involves offloading model parameters and activation tensors from GPU to CPU memory when they are not actively used. Although this dynamic offloading and subsequent reloading introduce additional data transfer time, it can greatly reduce the GPU memory footprint, allowing larger models to be trained even when GPU memory is a limiting factor.

### Using Memory-Friendly Optimizers  
- **Choosing Optimizers:** Opt for optimizers that consume less extra memory (e.g., SGD), even though they might affect model performance compared to more complex optimizers.
- **Optimizer Computation Modes:** Different computational strategies (such as for-loop, for-each, and fused implementations) can also impact both memory usage and training performance. 

### Use distributed methods to reduce video memory usage
There are many ways to use distributed systems to compress video memory. Among them, the method with the lowest threshold is FSDP. FSDP has a wide range of applications and can automatically split model parameters without much manual tuning. It also does not have many limitations in other model parallelism.

---

## Experimental Records

### Experimental Setup

- **CPU:** AMD EPYC 7713 64-Core Processor  
- **GPU:** 2 × A40 PCIe 48GB  
- **Training Code:** Based on the minGPT project ([minGPT on GitHub](https://github.com/karpathy/minGPT))  
- **Model:** gpt2-large  
- **Training Samples:** 10240

### Experimental Data

| Experiments Setting                                  | GPU Memory Peak | Training Latency |
| ---------------------------------------------------- | --------------- | ---------------- |
| Benchmark (batch size = 32)                          | 11846 MB        | 80.6 s           |
| Benchmark (batch size = 64)                          | 14338 MB        | 56.8 s           |
| Benchmark (batch size = 128)                         | 17022 MB        | 45.5 s           |
| + Cross-Batch Gradient Accumulation (32 × 4 = 128)     | 11736 MB        | 54.8 s           |
| + On-the-Fly Forward Tensor Recomputing              | 11500 MB        | 69.4 s           |
| + Optimizer for-loop Mode                            | 8786 MB         | 70.2 s           |
| + Dual-GPU FSDP Training                             | 7004 MB         | 74.9 s           |
