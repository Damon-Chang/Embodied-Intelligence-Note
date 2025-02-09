# DeekSeek系列
## DeepSeek-V3
**论文精读**

***
### 介绍
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

#### Model Summary
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
**模型下载**
<div align="center">

| **Model** | **#Total Params** | **#Activated Params** | **Context Length** | **Download** |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| DeepSeek-V3-Base | 671B | 37B | 128K   | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base)   |
| DeepSeek-V3   | 671B | 37B |  128K   | [🤗 Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3)   |

</div>

> [!NOTE]
> Hugging Face上DeepSeek-V3模型的总大小685B，包括主模型权重的671B和多令牌预测（MTP）模块权重的14B。**