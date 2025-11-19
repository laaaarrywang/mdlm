# Python、PyTorch 和 Lightning 语法学习笔记

本文档总结了在阅读 MDLM 项目代码过程中学习到的 Python、PyTorch 和 PyTorch Lightning 的关键语法和概念。

---

## 目录

1. [Python 基础语法](#1-python-基础语法)
2. [PyTorch 张量操作](#2-pytorch-张量操作)
3. [面向对象编程](#3-面向对象编程)
4. [PyTorch Lightning 框架](#4-pytorch-lightning-框架)
5. [TorchMetrics 指标系统](#5-torchmetrics-指标系统)
6. [MDLM 项目特定概念](#6-mdlm-项目特定概念)
7. [高级 Python 特性](#7-高级-python-特性)
8. [Lightning 生命周期钩子详解](#8-lightning-生命周期钩子详解)

---

## 1. Python 基础语法

### 1.1 解包操作符 `*`

**作用**：将元组、列表等可迭代对象展开成独立的位置参数。

```python
# 元组解包
x.shape = (3, 4)
*x.shape  # 展开为 3, 4

# 在函数调用中使用
x.view(*x.shape)  # 等价于 x.view(3, 4)

# 更多例子
x.shape = (2, 3, 4, 5)
x.view(*x.shape)  # 等价于 x.view(2, 3, 4, 5)
```

**对比其他语言**：
- JavaScript: `...args`
- Python: `*args`

**重要提醒**：`*x.shape` 不是"返回值"，而是在函数调用时的解包操作。

---

### 1.2 元组操作

```python
# 单元素元组（注意逗号）
(1,)      # 元组，包含一个元素
(1)       # 不是元组，是整数 1

# 元组复制
(1,) * 3  # 结果：(1, 1, 1)
(1, 2) * 2  # 结果：(1, 2, 1, 2)

# 实际应用：在 _unsqueeze 函数中
def _unsqueeze(x, reference):
    return x.view(
        *x.shape,  # 原始维度
        *((1,) * (len(reference.shape) - len(x.shape)))  # 添加的维度
    )

# 例如：x.shape=(3,4), reference.shape=(2,3,4,5)
# (1,) * (4-2) = (1,) * 2 = (1, 1)
# 最终：x.view(3, 4, 1, 1)
```

---

### 1.3 dataclass 装饰器

```python
from dataclasses import dataclass

@dataclass
class Loss:
    loss: torch.FloatTensor       # 标量损失
    nlls: torch.FloatTensor       # 每个 token 的 NLL
    token_mask: torch.FloatTensor # 有效 token 掩码
```

**自动生成**：
- `__init__()` 方法
- `__repr__()` 方法
- `__eq__()` 方法

**使用**：
```python
loss_obj = Loss(
    loss=torch.tensor(2.5),
    nlls=torch.tensor([2.3, 2.1]),
    token_mask=torch.tensor([1.0, 1.0])
)
print(loss_obj.loss)  # 访问属性
```

---

## 2. PyTorch 张量操作

### 2.1 view() 方法

**作用**：改变张量的形状，但不改变数据。

```python
x = torch.randn(2, 3, 4)  # shape: [2, 3, 4]
y = x.view(2, 12)          # shape: [2, 12]
z = x.view(6, 4)           # shape: [6, 4]
```

**动态形状**：
```python
x.view(*x.shape, 1, 1)  # 在末尾添加两个维度
```

---

### 2.2 张量的维度

以 logits 为例：

```python
logits.shape = [batch_size, sequence_length, vocab_size]
#               ^^^^^^^^^^  ^^^^^^^^^^^^^^^  ^^^^^^^^^^
#               批次大小     序列长度          词汇表大小

# 具体例子
logits.shape = [4, 1024, 50258]

# 访问
logits[0, 1, 1234]
#      ↑  ↑  ↑
#      |  |  └─ 词汇表索引 1234 的 token 的对数概率
#      |  └──── 位置 1（序列中的第二个 token）
#      └─────── 样本 0（批次中的第一个句子）
```

**实际含义**：
- **维度 0（batch_size）**：一次处理多少个样本
- **维度 1（sequence_length）**：每个句子的 token 数量
- **维度 2（vocab_size）**：每个位置预测每个 token 的对数概率

---

### 2.3 张量索引操作

```python
# 布尔索引
xt = torch.tensor([1, 50257, 3, 4])  # [token, MASK, token, token]
mask_indices = (xt == 50257)          # [False, True, False, False]

# 高级索引
logits = torch.randn(4, 1024, 50258)

# 1. 禁止预测 mask token
logits[:, :, self.mask_index] += self.neg_infinity
#      ↑  ↑  ^^^^^^^^^^^^^^^^
#      |  |  └─ 所有样本、所有位置的 mask token

# 2. 沿特定维度操作
logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
#                                         ^^^^^^
#                                         沿着 vocab_size 维度求和
```

---

## 3. 面向对象编程

### 3.1 继承与方法重写

```python
# 基类
class MeanMetric:
    def __init__(self):
        self.mean_value = 0.0
        self.weight = 0.0

    def update(self, value, weight=1.0):
        self.mean_value += value * weight
        self.weight += weight

    def compute(self):
        return self.mean_value / self.weight

# 子类：只重写 compute()
class BPD(MeanMetric):
    def compute(self):
        # 从 nats 转换为 bits
        return self.mean_value / self.weight / LOG2

class Perplexity(MeanMetric):
    def compute(self):
        # 取指数
        return torch.exp(self.mean_value / self.weight)
```

**关键点**：
- ✅ 继承了父类的 `__init__()` 和 `update()`
- ✅ 只重写了 `compute()` 方法
- ✅ 数据累积方式相同，计算方式不同

---

### 3.2 空类（占位符类）

```python
class NLL(torchmetrics.aggregation.MeanMetric):
    pass  # 什么都不做，只是语义化命名
```

**作用**：
1. **语义化**：`NLL` 比 `MeanMetric` 更具表达力
2. **继承关系**：为 BPD 和 Perplexity 提供共同祖先
3. **扩展性**：未来可以在此添加共同逻辑

**对比**：
```python
# 不够清晰
loss_metric = MeanMetric()

# 更清晰
nll_metric = NLL()
```

---

### 3.3 __init__() 的继承规则

**规则**：子类没有定义 `__init__()` 时，自动使用父类的。

```python
class BPD(NLL):
    def compute(self):  # 只重写方法
        return self.mean_value / self.weight / LOG2
    # 没有 __init__，自动继承 MeanMetric.__init__()
```

**何时需要写 __init__()**：
```python
# ✅ 需要（添加新属性）
class CustomMetric(MeanMetric):
    def __init__(self, num_dimensions):
        super().__init__()  # 调用父类初始化
        self.num_dimensions = num_dimensions  # 新属性
```

---

## 4. PyTorch Lightning 框架

### 4.1 save_hyperparameters()

**作用**：自动保存传入 `__init__()` 的参数到 checkpoint 中。

```python
class Diffusion(L.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.save_hyperparameters()  # 自动保存 config 和 tokenizer

        # 之后可以通过 self.hparams.config 访问
```

**保存的内容**：
- ✅ 函数参数：`config`, `tokenizer`
- ❌ 派生属性：`self.vocab_size`, `self.sampler`（不保存）

**访问方式**：
```python
self.hparams.config        # ✅ 传入的参数
self.hparams.vocab_size    # ❌ AttributeError（派生属性）
```

**处理不可序列化对象**：
```python
def __init__(self, config, tokenizer):
    super().__init__()
    # tokenizer 无法序列化，需要忽略
    self.save_hyperparameters(ignore=['tokenizer'])
    self.tokenizer = tokenizer  # 手动保存
```

---

### 4.2 Lightning 训练循环

```python
class Diffusion(L.LightningModule):
    def training_step(self, batch, batch_idx):
        """每个训练步调用一次"""
        loss = self._compute_loss(batch, prefix='train')
        self.log('train/loss', loss)  # 记录损失
        return loss  # 返回损失用于反向传播

    def validation_step(self, batch, batch_idx):
        """每个验证步调用一次"""
        loss = self._compute_loss(batch, prefix='val')
        return loss

    def on_validation_epoch_start(self):
        """验证 epoch 开始前调用"""
        self.valid_metrics.reset()

    def on_validation_epoch_end(self):
        """验证 epoch 结束后调用"""
        results = self.valid_metrics.compute()
        print(f"Validation: {results}")
```

**生命周期**：
1. `on_train_epoch_start()`
2. `training_step()` × N
3. `on_train_epoch_end()`
4. `on_validation_epoch_start()`
5. `validation_step()` × M
6. `on_validation_epoch_end()`

---

### 4.3 日志记录

```python
# 记录单个值
self.log('train/loss', loss, on_step=True, on_epoch=False)
#        ^^^^^^^^^^^^  ^^^^  ^^^^^^^^^^^^  ^^^^^^^^^^^^^
#        键名          值    每步记录      epoch平均

# 记录字典
self.log_dict(
    self.train_metrics,  # MetricCollection
    on_step=False,       # 不在每步记录
    on_epoch=True,       # 在 epoch 结束记录
    sync_dist=True       # 多 GPU 同步
)
```

**参数说明**：
- `on_step=True`：每个 step 记录一次
- `on_epoch=True`：每个 epoch 结束时记录平均值
- `sync_dist=True`：分布式训练时同步各 GPU 的值

---

## 5. TorchMetrics 指标系统

### 5.1 MeanMetric 基础

```python
from torchmetrics.aggregation import MeanMetric

metric = MeanMetric()

# 累积数据
metric.update(2.5, weight=10)  # 累积 2.5×10
metric.update(3.0, weight=20)  # 累积 3.0×20

# 计算加权平均
result = metric.compute()  # (2.5×10 + 3.0×20) / (10+20) = 2.83

# 重置
metric.reset()
```

**内部状态**：
- `mean_value`：累积的加权和
- `weight`：累积的权重总和

---

### 5.2 MetricCollection

**作用**：统一管理多个指标。

```python
metrics = torchmetrics.MetricCollection({
    'nll': NLL(),
    'bpd': BPD(),
    'ppl': Perplexity(),
})

# 一次性更新所有指标
metrics.update(loss, n_tokens)

# 计算所有指标
results = metrics.compute()  # {'nll': 2.2, 'bpd': 3.2, 'ppl': 9.2}

# 重置所有指标
metrics.reset()
```

---

### 5.3 clone() 方法

**作用**：创建独立副本，添加前缀/后缀。

```python
base_metrics = MetricCollection({'nll': NLL(), 'bpd': BPD()})

# 克隆出三份独立的副本
train_metrics = base_metrics.clone(prefix='train/')
val_metrics = base_metrics.clone(prefix='val/')
test_metrics = base_metrics.clone(prefix='test/')

# 各自独立更新
train_metrics.update(train_loss, train_tokens)
val_metrics.update(val_loss, val_tokens)

# 输出带前缀的键名
train_results = train_metrics.compute()
# {'train/nll': 2.3, 'train/bpd': 3.3}

val_results = val_metrics.compute()
# {'val/nll': 2.1, 'val/bpd': 3.0}
```

**为什么需要 clone()**：
- ❌ 共享同一个 metrics 会混合训练和验证数据
- ✅ 每个阶段独立的 metrics 避免污染

---

### 5.4 不同指标的 update() 参数

| Metric 类型 | update() 签名 | 参数说明 |
|------------|--------------|---------|
| **MeanMetric** | `update(value, weight=1.0)` | value: 值<br>weight: 权重 |
| **Accuracy** | `update(preds, target)` | preds: 预测<br>target: 标签 |
| **NLL/BPD/Perplexity** | `update(value, weight)` | 继承自 MeanMetric |

**MetricCollection 的限制**：
- ✅ 所有子指标必须接受**相同的参数**
- ❌ 如果参数不同，需要分别调用

```python
# ✅ 可行（参数相同）
metrics = MetricCollection({'nll': NLL(), 'bpd': BPD()})
metrics.update(loss, weight)  # 同时更新

# ❌ 不可行（参数不同）
metrics = MetricCollection({'loss': MeanMetric(), 'acc': Accuracy()})
# metrics.update(???)  # 无法同时满足两者
```

---

### 5.5 实际使用示例

```python
class Diffusion(L.LightningModule):
    def __init__(self):
        metrics = MetricCollection({
            'nll': NLL(),
            'bpd': BPD(),
            'ppl': Perplexity(),
        })

        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')

    def _compute_loss(self, batch, prefix):
        losses = self._loss(batch['input_ids'])

        # 选择对应的 metrics
        if prefix == 'train':
            self.train_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.train_metrics
        elif prefix == 'val':
            self.valid_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.valid_metrics

        # 记录到日志
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return losses.loss
```

---

## 6. MDLM 项目特定概念

### 6.1 Mask Token 的处理

```python
# 确保有 mask token 可用
if not hasattr(self.tokenizer, 'mask_token') or self.tokenizer.mask_token is None:
    # 情况1：tokenizer 没有 mask（如 GPT-2）
    self.mask_index = self.vocab_size  # 使用新索引
    self.vocab_size += 1                # 扩展词汇表
else:
    # 情况2：tokenizer 已有 mask（如 BERT）
    self.mask_index = self.tokenizer.mask_token_id
```

**作用**：
- GPT-2：动态添加 mask token（50257 → 50258）
- BERT：使用内置 mask token（103）

---

### 6.2 三种参数化方式

| 参数化 | 全称 | 特点 |
|--------|------|------|
| **SUBS** | SUBStitution | 本论文方法，简化为 MLM loss |
| **D3PM** | Discrete Denoising Diffusion | 吸收态扩散模型 |
| **SEDD** | Score Entropy Discrete Diffusion | 基于分数熵 |

---

### 6.3 subs_masking 参数

**作用**：在 D3PM 参数化中禁止预测 `[MASK]` token。

```python
def _d3pm_parameterization(self, logits):
    if self.subs_masking:
        # 将 mask token 的 logit 设为 -∞
        logits[:, :, self.mask_index] += self.neg_infinity

    # 归一化
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return logits
```

**效果**：
- `subs_masking=False`：模型可以预测任何 token（包括 mask）
- `subs_masking=True`：模型不能预测 mask，必须预测实际词汇

**理解**：
```python
# 输入
xt = ["The", "[MASK]", "is", "[MASK]"]

# subs_masking=False（允许预测 mask）
# 可能输出：["The", "cat", "is", "[MASK]"]  # 位置3仍是 mask

# subs_masking=True（禁止预测 mask）
# 必须输出：["The", "cat", "is", "cute"]  # 所有位置必须是实际词汇
```

---

### 6.4 SUBS vs D3PM 的关键区别

```python
# SUBS 参数化
def _subs_parameterization(self, logits, xt):
    # 1. 禁止预测 mask
    logits[:, :, self.mask_index] += self.neg_infinity

    # 2. 归一化
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    # 3. 关键！非 mask 位置强制保持不变
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

# D3PM 参数化（subs_masking=True）
def _d3pm_parameterization(self, logits):
    if self.subs_masking:
        logits[:, :, self.mask_index] += self.neg_infinity

    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    # 注意：没有强制保持非 mask 位置的逻辑
    return logits
```

**对比**：

| 特性 | SUBS | D3PM (subs_masking=True) |
|------|------|-------------------------|
| 禁止预测 mask | ✅ | ✅ |
| 非 mask 位置保持不变 | ✅ | ❌ |
| mask 位置预测 | 必须是实际 token | 必须是实际 token |

---

### 6.5 Backbone 模型类型

| Backbone | 类型 | 特点 |
|----------|------|------|
| **DIT** | Diffusion Transformer | 标准 Transformer，适合扩散模型 |
| **DiMamba** | Diffusion Mamba | State Space Model，线性复杂度 |
| **AR** | Autoregressive | GPT 风格，作为 baseline |
| **HF_DIT** | HuggingFace 预训练模型 | 从 Hub 加载 |

---

## 附录：常见问题

### Q1: 为什么 BPD 要除以 LOG2？

**答案**：对数换底公式。

```python
# NLL 使用自然对数（以 e 为底）
nll = ln(P)  # nats

# BPD 使用以 2 为底的对数
bpd = log₂(P)  # bits

# 换底公式：log₂(x) = ln(x) / ln(2)
bpd = nll / LOG2
```

### Q2: Perplexity 为什么要取 exp？

**答案**：Perplexity 是 NLL 的指数形式。

```python
nll = -log(P)
perplexity = exp(nll) = exp(-log(P)) = 1/P
```

**解释**："困惑度为 10" 意味着模型在每个位置平均在 10 个选项中犹豫。

### Q3: 为什么需要三种指标（NLL、BPD、Perplexity）？

**答案**：不同使用场景。

| 指标 | 单位 | 使用场景 |
|------|------|----------|
| NLL | nats | 训练优化、损失函数 |
| BPD | bits | 论文对比、压缩性能 |
| Perplexity | 无 | 语言模型评估、可解释性 |

---

## 7. 高级 Python 特性

### 7.1 itertools.chain()

**作用**：将多个可迭代对象连接成一个。

```python
import itertools

# 基本用法
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = itertools.chain(list1, list2)

for item in combined:
    print(item)
# 输出：1, 2, 3, 4, 5, 6

# 连接不同类型
letters = ['a', 'b']
numbers = (1, 2)
combined = itertools.chain(letters, numbers)
list(combined)  # ['a', 'b', 1, 2]
```

---

### 7.2 在 PyTorch 中的使用

**问题场景**：需要对多个模型的参数进行相同操作。

```python
# 项目中有两组参数
backbone_params = self.backbone.parameters()  # 主干网络参数
noise_params = self.noise.parameters()        # 噪声调度参数

# ❌ 错误做法
self.ema.update(self.backbone.parameters())
self.ema.update(self.noise.parameters())  # 会覆盖前面的

# ✅ 正确做法：合并为一个迭代器
all_params = itertools.chain(
    self.backbone.parameters(),
    self.noise.parameters()
)
self.ema.update(all_params)  # 一次性更新所有参数
```

---

### 7.3 项目中的三个使用场景

#### 场景 1：EMA 初始化

```python
# __init__() 中
self.ema = ExponentialMovingAverage(
    itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()
    ),
    decay=0.9999
)
```

#### 场景 2：优化器初始化

```python
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        itertools.chain(
            self.backbone.parameters(),
            self.noise.parameters()
        ),
        lr=self.config.optim.lr,
        # ...
    )
```

#### 场景 3：EMA 更新

```python
def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
        self.ema.update(itertools.chain(
            self.backbone.parameters(),
            self.noise.parameters()
        ))
```

---

### 7.4 chain() 的优势

#### 优势 1：惰性求值（Lazy Evaluation）

```python
# chain() 创建迭代器，不立即复制数据
combined = itertools.chain(
    self.backbone.parameters(),  # 100M 个参数
    self.noise.parameters()       # 1K 个参数
)
# 内存占用：O(1)，只是迭代器

# 对比：列表拼接
combined = list(backbone_params) + list(noise_params)
# 内存占用：O(n)，创建了副本
```

#### 优势 2：代码清晰

```python
# ✅ 明确表达"合并两组参数"
all_params = itertools.chain(backbone_params, noise_params)

# ❌ 需要理解列表操作
all_params = list(backbone_params) + list(noise_params)
```

---

## 8. Lightning 生命周期钩子详解

### 8.1 钩子方法 vs 覆盖方法

**关键区别**：理解什么时候需要 `super()`。

| 方法类型 | 父类实现 | 需要 super() | 示例 |
|---------|---------|-------------|------|
| **覆盖方法** | 有实际逻辑 | ✅ 必需 | `optimizer_step` |
| **钩子方法** | 空实现（pass） | ❌ 不需要 | `on_save_checkpoint` |
| **抽象方法** | 必须实现 | ❌ 不需要 | `training_step` |

---

### 8.2 optimizer_step() - 需要 super()

```python
# Lightning 父类
class LightningModule:
    def optimizer_step(self, epoch, batch_idx, optimizer, ...):
        """执行优化器更新"""
        optimizer.step()      # 真正的更新逻辑
        optimizer.zero_grad() # 清空梯度

# 项目中的实现
class Diffusion(L.LightningModule):
    def optimizer_step(self, *args, **kwargs):
        # 必须调用 super()，否则参数不会更新
        super().optimizer_step(*args, **kwargs)

        # 然后添加额外逻辑（更新 EMA）
        if self.ema:
            self.ema.update(itertools.chain(
                self.backbone.parameters(),
                self.noise.parameters()
            ))
```

**如果不用 super() 会怎样？**

```python
# ❌ 错误示例
def optimizer_step(self, *args, **kwargs):
    if self.ema:
        self.ema.update(...)
    # 没有调用 super()
    # 结果：optimizer.step() 永远不会被调用
    #      模型参数永远不会更新！
```

---

### 8.3 on_save_checkpoint() - 不需要 super()

**关键理解**：`checkpoint['state_dict']` 不是父类方法添加的，而是 **Trainer** 添加的！

```python
# Lightning 父类（空实现）
class LightningModule:
    def on_save_checkpoint(self, checkpoint):
        pass  # 什么都不做

# Trainer 的保存流程
class Trainer:
    def save_checkpoint(self, model, path):
        checkpoint = {}

        # 1. Trainer 添加标准内容
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['optimizer_states'] = [...]
        checkpoint['epoch'] = self.current_epoch
        # ...

        # 2. 调用模型的钩子（让模型添加自定义内容）
        model.on_save_checkpoint(checkpoint)
        #                        ^^^^^^^^^^
        #           此时已经包含了 state_dict

        # 3. 保存
        torch.save(checkpoint, path)
```

**项目中的实现**：

```python
def on_save_checkpoint(self, checkpoint):
    # checkpoint 传入时已包含（Trainer 添加的）：
    # ✓ checkpoint['state_dict']
    # ✓ checkpoint['optimizer_states']
    # ✓ checkpoint['loops']

    # 只需添加 Lightning 不知道的内容
    if self.ema:
        checkpoint['ema'] = self.ema.state_dict()

    # 修正 batch 计数（因为梯度累积）
    checkpoint['loops']['fit_loop'][...] = corrected_value
```

**为什么不需要 super()？**

```python
# 即使写了也没用
def on_save_checkpoint(self, checkpoint):
    super().on_save_checkpoint(checkpoint)  # 调用 pass
    checkpoint['ema'] = self.ema.state_dict()

# 等价于
def on_save_checkpoint(self, checkpoint):
    checkpoint['ema'] = self.ema.state_dict()
```

---

### 8.4 on_save_checkpoint 的覆盖原则

#### 原则 1：只添加 Lightning 不知道的内容

```python
def on_save_checkpoint(self, checkpoint):
    # ✅ 添加自定义状态
    if self.ema:
        checkpoint['ema'] = self.ema.state_dict()

    # ❌ 不要重复保存 Lightning 已处理的
    # checkpoint['state_dict'] = self.state_dict()  # Lightning 已做
```

#### 原则 2：修改现有键时要小心

```python
def on_save_checkpoint(self, checkpoint):
    # ✅ 可以修正 Lightning 的默认值
    checkpoint['loops']['fit_loop'][...] = corrected_value

    # ⚠️ 需要理解 Lightning 内部结构
```

#### 原则 3：保持对称性

```python
# 保存时
def on_save_checkpoint(self, checkpoint):
    checkpoint['ema'] = self.ema.state_dict()

# 加载时（必须对应）
def on_load_checkpoint(self, checkpoint):
    self.ema.load_state_dict(checkpoint['ema'])
```

#### 原则 4：处理可选内容

```python
def on_save_checkpoint(self, checkpoint):
    if self.ema:
        checkpoint['ema'] = self.ema.state_dict()

def on_load_checkpoint(self, checkpoint):
    if self.ema and 'ema' in checkpoint:
        self.ema.load_state_dict(checkpoint['ema'])
```

---

### 8.5 三个关键生命周期钩子

#### 8.5.1 on_load_checkpoint() - 恢复训练状态

```python
def on_load_checkpoint(self, checkpoint):
    # 1. 恢复 EMA 参数
    if self.ema:
        self.ema.load_state_dict(checkpoint['ema'])

    # 2. 恢复训练进度（用于断点续训）
    self.fast_forward_epochs = checkpoint['loops'][
        'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
        'fit_loop']['epoch_loop.batch_progress'][
            'current']['completed']
```

**作用**：
- 恢复 EMA 的影子参数（平滑版本的模型权重）
- 恢复训练进度信息（哪个 epoch、哪个 batch）

---

#### 8.5.2 on_save_checkpoint() - 保存额外状态

```python
def on_save_checkpoint(self, checkpoint):
    # 1. 保存 EMA 参数
    if self.ema:
        checkpoint['ema'] = self.ema.state_dict()

    # 2. 修正 batch 进度（梯度累积导致的问题）
    checkpoint['loops']['fit_loop'][
        'epoch_loop.batch_progress']['total']['completed'] = \
            optimizer_steps * self.trainer.accumulate_grad_batches

    # 3. 保存数据加载器随机状态
    if hasattr(self.trainer.train_dataloader.sampler, 'state_dict'):
        checkpoint['sampler']['random_state'] = \
            self.trainer.train_dataloader.sampler.state_dict()
```

**为什么需要修正 batch 进度？**

```python
# 梯度累积场景
global_batch_size = 512
batch_size_per_gpu = 64
num_gpus = 4
accumulate_grad_batches = 2  # 累积 2 步

# Lightning 默认统计：处理了 2 个 mini-batch
# 实际应该是：2 × accumulate_grad_batches = 4 个全局 batch

# 因此需要修正
actual_batches = optimizer_steps * accumulate_grad_batches
```

---

#### 8.5.3 on_train_start() - 设置训练环境

```python
def on_train_start(self):
    # 1. 将 EMA 参数移到 GPU
    if self.ema:
        self.ema.move_shadow_params_to_device(self.device)

    # 2. 判断是否分布式训练
    distributed = (
        self.trainer._accelerator_connector.use_distributed_sampler
        and self.trainer._accelerator_connector.is_distributed)

    # 3. 创建支持断点续训的数据加载器
    if distributed:
        sampler_cls = FaultTolerantDistributedSampler
    else:
        sampler_cls = RandomFaultTolerantSampler

    # 4. 如果是断点续训，恢复数据加载器状态
    if (self.fast_forward_epochs is not None and
        self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
            'epoch': self.fast_forward_epochs,
            'counter': self.fast_forward_batches * batch_size
        })
```

**作用**：
- 设备管理：将 EMA 参数移到正确的设备
- 数据加载器：创建支持断点续训的 sampler
- 断点续训：从正确的数据位置继续

---

### 8.6 完整的训练流程

#### 正常训练

```
trainer.fit(model)
  ↓
1. __init__()
  ↓
2. on_train_start()
   └─ EMA 移到 GPU
   └─ 创建容错 sampler
  ↓
3. training_step() × N
  ↓
4. on_save_checkpoint()  ← 定期保存
  ↓
继续...
```

#### 断点续训

```
trainer.fit(model, ckpt_path='last.ckpt')
  ↓
1. __init__()
  ↓
2. on_load_checkpoint()
   └─ 恢复 EMA 参数
   └─ fast_forward_epochs = 5
   └─ fast_forward_batches = 1234
  ↓
3. on_train_start()
   └─ EMA 移到 GPU
   └─ sampler.load_state_dict({epoch: 5, counter: 78976})
   └─ 数据加载器从第 78977 个样本开始
  ↓
4. training_step() × N  ← 从中断处继续
```

---

### 8.7 三个钩子的协作总结

| 钩子 | 调用时机 | 主要作用 | 关键操作 |
|------|---------|---------|---------|
| **on_load_checkpoint** | 加载 ckpt 时 | 恢复额外状态 | 恢复 EMA + 训练进度 |
| **on_save_checkpoint** | 保存 ckpt 时 | 添加额外信息 | 保存 EMA + 修正进度 + 数据状态 |
| **on_train_start** | 训练开始前 | 设置训练环境 | EMA 移设备 + 恢复数据加载器 |

---

## 9. 项目初始化详解

### 9.1 初始化流程中的关键组件

```python
def __init__(self, config, tokenizer):
    super().__init__()
    self.save_hyperparameters()

    # 1. 生成评估相关
    self.gen_ppl_metric = Perplexity()
    self.eval_model_tokenizer = transformers.AutoTokenizer.from_pretrained(
        self.gen_ppl_eval_model_name_or_path)

    # 2. 噪声调度
    self.noise = noise_schedule.get_noise(self.config, dtype=self.dtype)

    # 3. EMA（指数移动平均）
    if self.config.training.ema > 0:
        self.ema = ExponentialMovingAverage(
            itertools.chain(
                self.backbone.parameters(),
                self.noise.parameters()
            ),
            decay=self.config.training.ema
        )

    # 4. 常用超参数
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.neg_infinity = -1000000.0

    # 5. 断点续训参数
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
```

---

### 9.2 组件详解

#### 生成困惑度评估

```python
self.gen_ppl_metric = Perplexity()
self.eval_model_tokenizer = transformers.AutoTokenizer.from_pretrained(
    'gpt2-large'  # 或其他强大的 AR 模型
)
```

**作用**：
- 用**另一个预训练模型**（如 GPT-2）评估生成样本的质量
- 困惑度越低 → 生成文本越自然

**流程**：
```python
# 1. 扩散模型生成样本
generated_text = diffusion_model.sample()

# 2. 用 GPT-2 计算困惑度
gpt2_perplexity = eval_model.compute_perplexity(generated_text)
```

---

#### 噪声调度

```python
self.noise = noise_schedule.get_noise(self.config, dtype=self.dtype)
```

**作用**：控制扩散过程中的噪声添加。

```python
# 噪声调度决定在时间 t 时的噪声量
t=0.0 → "The cat is cute"  # 0% 噪声
t=0.3 → "The cat is [M]"   # 30% mask
t=0.7 → "The [M] [M] [M]"  # 70% mask
t=1.0 → "[M] [M] [M] [M]"  # 100% mask
```

**支持的类型**：
- `loglinear`：对数线性调度
- `cosine`：余弦调度
- `geometric`：几何调度

---

#### EMA（指数移动平均）

```python
if self.config.training.ema > 0:
    self.ema = ExponentialMovingAverage(
        itertools.chain(
            self.backbone.parameters(),
            self.noise.parameters()
        ),
        decay=0.9999
    )
```

**作用**：维护模型参数的平滑版本。

**更新公式**：
```python
shadow_param = decay * shadow_param + (1 - decay) * current_param
# decay = 0.9999
```

**使用流程**：
```python
# 训练时
optimizer.step()
self.ema.update(model.parameters())

# 验证时
self.ema.store(model.parameters())      # 保存当前参数
self.ema.copy_to(model.parameters())    # 用 EMA 参数替换
val_loss = validate(model)              # 用平滑参数验证
self.ema.restore(model.parameters())    # 恢复原参数
```

---

#### 常量和超参数

```python
self.lr = self.config.optim.lr                    # 学习率
self.sampling_eps = self.config.training.sampling_eps  # 采样最小时间
self.neg_infinity = -1000000.0                    # 负无穷常量
```

**neg_infinity 的使用**：
```python
# 禁止预测 mask token
logits[:, :, self.mask_index] += self.neg_infinity
# log(P(mask)) = -1000000 ≈ -∞
# P(mask) = exp(-1000000) ≈ 0
```

---

## 参考资源

- [PyTorch 官方文档](https://pytorch.org/docs/)
- [PyTorch Lightning 文档](https://lightning.ai/docs/pytorch/stable/)
- [TorchMetrics 文档](https://torchmetrics.readthedocs.io/)
- [MDLM 论文](https://arxiv.org/abs/2406.07524)
- [MDLM GitHub](https://github.com/s-sahoo/mdlm)

---

---

**最后更新**：2025-01-18（第二次更新）

**本次更新内容**：
- 新增第 7 章：高级 Python 特性（itertools.chain 详解）
- 新增第 8 章：Lightning 生命周期钩子详解（on_save_checkpoint、on_load_checkpoint、on_train_start）
- 新增第 9 章：项目初始化详解（EMA、噪声调度、生成评估）
- 详细解释了 super() 的使用时机和原则
- 补充了钩子方法 vs 覆盖方法的区别
- 添加了断点续训的完整流程说明
