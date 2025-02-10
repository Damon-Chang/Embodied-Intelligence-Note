# DeekSeek系列

- [DeepSeek-V3](#deepseek-v3)
  - [介绍](#介绍)
  - [精读](#精读)
- [DeepSeek R1](#deepseek-r1)

## DeepSeek-V3 <a href="#deekseek系列">🔝</a>

---
### 介绍 <a href="#deekseek系列">🔝</a>
- **名称**：DeepSeek-V3
- **类型**：Mixture-of-Experts（MoE）大语言模型。Mixture-of-Experts (MoE) language model，即混合专家语言模型，是一种先进的语言模型架构，通过整合多个专家模块提升性能，近年来在自然语言处理领域备受关注。
- **参数量**：671B，每个token激活37B的参数。
- **组件**
  - **MLA**：Multi-head Latend Attention；
  - **DeepSeekMoE**：DeepSeek专门开发的混合专家模型；
- **创新**
  - 首创一种**无辅助损失策略**（auxiliary-loss-free strateg）来应对负载均衡问题（load balancing）；
  - 设定**多词元训练目标**（ multi-token prediction training
objective），以实现更好的性能
- **训练资源**
  - **数据**：1.48万亿（14.8trillion）个多样且高质量的词元上对DeepSeek-V3进行预训练；
  - 随后历经**监督微调**与**强化学习**阶段，以充分发挥其性能。
- **表现**
  - 全面评估显示，DeepSeek-V3的表现超越了其他开源模型，达到了与领先的闭源模型相当的水平。
  - DeepSeek-V3完成全部训练仅需**278.8万个H800 GPU小时**；
  - 其训练过程极为稳定。在整个训练过程中，没有遇到任何无法挽回的**损失激增**（loss spikes）情况，也未进行任何**回滚**（rollbacks）操作。
![alt text](<截屏2025-02-09 12.07.46.png>)

#### 1.Model Summary
---
**结构：创新了负载均衡策略（Load Balancing Strategy）和训练目标（Training Objective）**
- 在DeepSeek-V2的高效架构之上，V3开创了负载均衡的无辅助损失策略，该策略最大限度地减少了因鼓励负载平衡而导致的性能下降；
- V3研究了一个多词元预测（Multi-TokenPrediction，MTP）目标并证明它有利于模型性能。它也可以用于推理加速的推测解码。

---
**预训练：迈向终极训练效率**
- V3设计了一个FP8混合精度训练框架，并首次在超大规模模型上验证了FP8训练的可行性和有效性；
- 通过算法、框架和硬件的协同设计，V3克服了跨节点MoE训练中的通信瓶颈，几乎实现了全computation-communication重叠。这显着提高了模型训练效率并降低了训练成本，使得能够在没有额外开销的情况下进一步扩展模型大小；
- 以仅2.664M H800 GPU小时的经济成本，我们在14.8T tokens上完成了DeepSeek-V3的预训练，产生了目前最强的开源基础模型。预训练后的后续训练阶段只需要0.1M GPU小时。

---
**后训练：DeepSeek-R1的知识提炼**
- V3引入了一种创新的方法论，将long-Chain-of-Thought（CoT）模型，特别是DeepSeek R1系列模型之一的推理能力提取到标准LLM中，尤其是DeepSeek-V3。我们的管道将R1的验证和反射模式优雅地整合到DeepSeek-V3中，并显着提高了其推理性能。同时，我们还保持对DeepSeek-V3输出样式和长度的控制。

---
#### 2.模型下载
<div align="center">

| **Model** | **#Total Params** | **#Activated Params** | **Context Length** | **Download** |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| DeepSeek-V3-Base | 671B | 37B | 128K   | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base)   |
| DeepSeek-V3   | 671B | 37B |  128K   | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3)   |

</div>

> [!NOTE]
> Hugging Face上DeepSeek-V3模型的总大小685B，包括主模型权重的671B和多词元预测（MTP）模块权重的14B。

#### 3.交互网站 & API平台
与 DeepSeek-V3 聊天在DeepSeek的[官方网址](chat.deepseek.com)

提供了OpenAI-Compatible API 在 DeepSeek 的[平台](platform.deepseek.com)

### 精读 <a href="#deekseek系列">🔝</a>
考虑到强大的模型性能和经济成本，DeepSeek-V3模型以前瞻性的视角，沿用**Multi-head Latend Attention（MLA）**实现高效推理，使用**DeepSeekMoE**实现经济高效的训练。这两种架构已在[DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2?utm_source=catalyzex.com)中得到验证，展示了它们在保持稳健模型性能的同时实现高效训练和推理的能力。
除了这两个基本结构，两个额外的策略被用于增强模型性能。DeepSeek-V3为负载平衡开创了一种**无辅助损失策略**，目的是最大限度地减少因鼓励负载平衡而对模型性能产生的不利影响。其次，DeepSeek-V3采用了**多token预测训练目标**，我们已经观察到它可以提高评估基准的整体性能。
为了高效训练，V3使用FP8混合精度训练，并全面优化训练框架。
> **低精度**训练已经成为高效训练的有希望的解决方案，其演变与硬件能力的进步密切相关。V3引入了FP8混合精读训练框架，并首次在超大规模模型上验证了其有效性。
> - 支持FP8计算和存储，实现了**加速训练**和**减少GPU内存使用**；
> - 对于训练框架，设计了**DualPipe算法**以实现高效的管道并行性，该算法具有更少的管道气泡，并通过**computation-communication重叠**隐藏了训练期间的大部分通信。
> 
这种重叠确保了随着模型的扩大，只要我们保持恒定的computation-to-communication比，我们仍然可以跨节点使用细粒度的专家（fine-grained experts），同时实现接近零的all-to-all全通信开销。
此外，V3还开发了高效的**跨节点all-to-all通信内核**，以充分利用InfiniBand（IB）和NVLink带宽。
此外，V3精心优化了内存赵勇，使得在不使用昂贵的张量并行性的情况下训练DeepSeek-V3成为了可能，结合这些努力，V3实现了高效训练效率。
> [!NOTE]
> 看下来DeepSeek确实花了很多心思在如何用有限的算力实现大模型训练上，无论是模型的结构上还是训练架构上，都做了很多优化，还有创新。这样的模型感觉起来确实更加智能。好厉害！

---

**1.预训练阶段：**
- 14.8T高质量多样的词元上进行训练，训练过程十分稳定，没有损失激增和回滚操作。

**2.两阶段的DeepSeek-V3上下文长度扩展**
- 第一阶段：最大上下文长度扩展到32K；
- 第二阶段：最大上下文长度扩展到128K。

**3.后训练阶段：**
- **有监督微调（Supervised Fine-Tuning，SFT）**；
- **强化学习（Reinforcement Learning，RL）**；
- 将DeepSeek-V3与人类喜好对齐，并进一步释放其潜力。

在后训练阶段，从DeepSeek-R1系列模型中提取推理能力，同时小心保持模型精度和生成长度之间的平衡。

![alt text](<截屏2025-02-10 12.32.16.png>)

在预训练阶段，在每万亿个令牌上训练 DeepSeek-V3 只需要 180KH800 GPU 小时，即在我们拥有 2048 个 H800 GPU 的集群上训练 3.7 天。因此，我们的预训练阶段在**不到两个月**的时间内完成，花费 2664KGPU 小时。加上上下文长度扩展的 119KGPU 小时和训练后的 5KGPU 小时，DeepSeek-V3 的完整训练仅花费 2.788MGPU 小时。假设 H800 GPU 的租赁价格为每 GPU 小时 2 美元，我们的总训练成本仅为 557.60 万美元。请注意，上述成本仅包括 DeepSeek-V3 的官方训练，不包括与架构、算法或数据的先前研究和消融实验相关的成本。
> 也蛮烧钱的，不过如此大模型在两个月的时间内训练完成，非常了不起！👍

---
#### 主要贡献
**结构：创新负载均衡策略和训练目标**
- **负载均衡策略**：在DeepSeek-V2的高效架构之上，开创了负载平衡的无辅助损失策略，该策略最大限度地减少了因鼓励负载平衡而导致的性能下降。
- **训练目标**：研究了多令牌预测（MTP）目标，并证明它有利于模型性能。它也可用于推理加速的推测解码。
- 
**预培训：迈向终极培训效率**
- V3设计了一个FP8混合精度训练框架，并首次在超大规模模型上验证了FP8训练的可行性和有效性；
- 通过算法、框架和硬件的协同设计，我们克服了跨节点MoE训练中的通信瓶颈，实现了近乎完全的计算-通信重叠。这显着提高了我们的训练效率，降低了训练成本，使我们能够在没有额外开销的情况下进一步扩大模型大小；
- 我们以仅2.664M H800 GPU小时的经济成本完成了14.8T令牌上DeepSeek-V3的预训练，产生了目前最强大的开源基础模型。预训练后的后续训练阶段只需要0.1MGPU小时。

**后训练：DeepSeek-R1的知识提炼**
- 引入了一种创新的方法论，将长思维链（long-Chain-of-Thought，CoT）模型，特别是DeepSeek R1系列模型之一的推理能力提炼到标准LLM中，尤其是DeepSeek-V3。我们的管道优雅地将R1的验证和反射模型结合到的V3中，显着提高其推理性能。同时，我们还保持对DeepSeek-V3输出样式和长度的控制。

**核心评估结果摘要**
- **知识：**
  1. 在MMLU、MMLU-Pro和GPQA等教育基准上，DeepSeek-V3的表现优于所有其他开源模型，在MMLU上达到88.5，在MMLU-Pro上达到75.9，在GPQA上达到59.1。它的性能可与GPT-4o和Claude-Sonnet-3.5等领先的闭源模型相媲美，缩小了该领域的开源和闭源模型之间的差距；
  2. 对于事实性基准，DeepSeek-V3在SimpleQA和中国SimpleQA的开源模型中都表现出卓越的性能。虽然它在英语事实知识（SimpleQA）方面落后于GPT-4o和Claude-Sonnet-3.5，但它在中文事实知识（中文SimpleQA）方面超越了这些模型，突出了其在中文事实知识方面的优势。

- **代码、数学和推理：**
  1. 在所有非长CoT开源和闭源模型中，DeepSeek-V3在数学相关基准测试上实现了最先进的性能。值得注意的是，它甚至在特定基准测试（如MATH-500）上的性能优于o1-preview，展示了其强大的数学推理能力；
  2. 在编码相关任务上，DeepSeek-V3成为编码竞争基准测试（如LiveCodeBench）的最佳模型，巩固了其在该领域的领先模型地位。对于工程相关任务，尽管DeepSeek-V3的性能略低于Claude-Sonnet-3.5，但它仍然远远超过所有其他模型，展示了其在各种技术基准测试中的竞争力。<a href="#deekseek系列">🔝</a>

---
#### 结构
DeepSeek-V3基本的结构特点是：多头潜在注意力（Multi-head Attention，MLA）用于高效推理，DeepSeekMoE用于高效经济训练。多令牌预测（Multi-Token Prediction，MTP）训练目标，我们观察到它可以提高评估基准的整体性能。对于其他未明确提及的小细节，DeepSeek-V3遵循DeepSeek-V2（DeepSeek-AI，2024c）的设置。
##### 基础结构
DeepSeel-V3的基础结构仍在Transformer结构内。为了高效推理和经济训练采用了V2中的MLA和DeepSeekMoE。与V2相比为DeepSeek-V3的DeepSeekMoE新增了无辅助损失的负载均衡（auxiliary-loss-free load balancing）策略，减轻由确保负载平衡的努力引起的性能下降。
![img](https://github.com/Damon-Chang/Embodied-Intelligence-Note/blob/main/figures/%E6%88%AA%E5%B1%8F2025-02-10%2014.19.30.png)
![alt text](<截屏2025-02-10 14.19.30.png>)


## DeepSeek R1 <a href="#deekseek系列">🔝</a>



