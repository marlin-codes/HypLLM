
## Hyperbolic Low-rank Fine-tuning for LLMs

### 1.Introduction
   Large language models (LLMs) have demonstrated remarkable performance on various tasks. However, it remains an open question whether the default Euclidean space is the most suitable choice for embedding tokens in LLMs. In this study, we first investigate the non-Euclidean characteristics of LLMs. 
   Our findings reveal that token frequency follows a power-law distribution, with high-frequency tokens clustering near the origin and low-frequency tokens positioned farther away. Additionally, token embeddings exhibit a high degree of hyperbolicity, indicating a latent tree-like structure in the embedding space. Building on the observation, we propose to efficiently fine-tune LLMs in hyperbolic space to better exploit the underlying complex structures.  However, we found that this fine-tuning in hyperbolic space cannot be achieved with naive application of exponential and logarithmic maps, when the embedding and weight matrices both reside in Euclidean space.
   To address this technique issue, we introduce a new method called hyperbolic low-rank efficient fine-tuning, HypLoRA, that performs low-rank adaptation directly on the hyperbolic manifold, avoiding the cancellation effect caused by the exponential and logarithmic maps, thus preserving the hyperbolic modeling capabilities. Through extensive experiments, we demonstrate that HypLoRA significantly enhances the performance of LLMs on reasoning tasks, particularly for complex reasoning problems. 

### 2.Power-law Distribution in Token Embedding

| ![GSM8K Token Frequency](./utils/results/figs_frequency/gsm8k/GSM8K_token_frequency_distribution.png)  | ![AQuA Token Frequency](./utils/results/figs_frequency/AQuA/AQuA_token_frequency_distribution.png)  | ![BoolQ Token Frequency](./utils/results/figs_frequency/boolq/BoolQ_token_frequency_distribution.png)  |
|:----------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------:|
| ![Math 10K Token Frequency](./utils/results/figs_frequency/math_10k/math_10k_token_frequency_distribution.png) | ![Math 50K Token Frequency](./utils/results/figs_frequency/math_50k/math_50k_token_frequency_distribution.png) | ![MAWPS Token Frequency](./utils/results/figs_frequency/mawps/MAWPS_token_frequency_distribution.png) |
| ![OpenBookQA Token Frequency](./utils/results/figs_frequency/openbookqa/OpenBookQA_token_frequency_distribution.png) | ![SVAMP Token Frequency](./utils/results/figs_frequency/SVAMP/SVAMP_token_frequency_distribution.png) | ![WinoGrande Token Frequency](./utils/results/figs_frequency/winogrande/WinoGrande_token_frequency_distribution.png) |

### 3. Hierarchical examples in Token Embedding

![img.png](./figs/numbers.png)

### 4. Frequency Distribution w.r.t. Norm

| ![AQuA Frequency vs Norm](./utils/results/figs_frequency_norm/AQuA/AQuA_binned_frequency_vs_norm.png)  | ![BoolQ Frequency vs Norm](./utils/results/figs_frequency_norm/boolq/boolq_binned_frequency_vs_norm.png)  | ![GSM8K Frequency vs Norm](./utils/results/figs_frequency_norm/gsm8k/GSM8K_binned_frequency_vs_norm.png)  |
|:-----------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|
| ![Math 10K Frequency vs Norm](./utils/results/figs_frequency_norm/math_10k/math_10k_binned_frequency_vs_norm.png) | ![Math 50K Frequency vs Norm](./utils/results/figs_frequency_norm/math_50k/math_50k_binned_frequency_vs_norm.png) | ![MAWPS Frequency vs Norm](./utils/results/figs_frequency_norm/mawps/MAWPS_binned_frequency_vs_norm.png) |
| ![OpenBookQA Frequency vs Norm](./utils/results/figs_frequency_norm/openbookqa/openbookqa_binned_frequency_vs_norm.png) | ![SVAMP Frequency vs Norm](./utils/results/figs_frequency_norm/SVAMP/SVAMP_binned_frequency_vs_norm.png) | ![WinoGrande Frequency vs Norm](./utils/results/figs_frequency_norm/winogrande/winogrande_binned_frequency_vs_norm.png) |



### 5.Core Code for Hyperbolic low-rank fine-tuning LLMs

```python
x = x.to(self.lora_A.weight.dtype)
x = self.lora_dropout(x)

x = self.padding_zero(x)

# (1) exponential map
x = self.lorentz_expmap0(x, self.k)

# (2) Transformation on the manifold
x_space = self.lora_A(x)
x_time = ((x_space ** 2).sum(dim=-1, keepdim=True) + self.k).sqrt()  #
x = torch.cat([x_time, x_space], dim=-1)  # cat the time value
x_space = self.lora_B(x)
x_time = ((x_space ** 2).sum(dim=-1, keepdim=True) + self.k).sqrt()
x = torch.cat([x_time, x_space], dim=-1)  # cat the time value

# (3) Logarithmic map
x = self.lorentz_logmap0(x, self.k)[..., 1:]
x = x * self.scaling
result += x

```

### 6. Running Scripts

To be updated in `example` folder.