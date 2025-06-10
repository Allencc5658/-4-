# 文档介绍

## 一 训练参数

**deepseek_lora_sft_bitsandbytes.yaml文件展示了参数设置**

QLoRA微调代码详细的参数解释如下：

1. 模型基础信息：

model_name_or_path: DeepSeek-R1-Distill-Qwen-7B：指定了要微调的基础模型的路径或名称。

quantization_bit: 4：表示将模型的权重进行量化的位数为 4 位。对应了QLoRA的量化步骤

2. 微调方法：

stage: sft：这里的sft代表“Supervised Fine-Tuning”（有监督微调），即使用有标注的数据对模型进行微调训练。

do_train: true：表示要进行训练操作。

finetuning_type: lora：使用“LoRA”（Low-Rank Adaptation）方法进行微调。LoRA 是一种高效的微调技术，它通过在原始模型的基础上添加一些低秩矩阵来实现对模型的微调，从而减少了训练的参数数量和计算成本。

lora_target: all：表示对模型的所有部分都应用 LoRA 微调。

3. 数据集相关：

dataset: jiakao：指定了使用的驾考数据集

template: deepseek：指定了数据的模板或格式与deepseek模型相匹配。这有助于将数据集转换为适合模型输入的格式。

cutoff_len: 1024：设置输入文本的截断长度为 1024。如果输入文本超过这个长度，会被截断以适应模型的处理能力。

max_samples: 100000：限制数据集中使用的最大样本数量为 100000。这是出于训练时间或资源限制的考虑。

overwrite_cache: true：这意味着每次运行时都会重新处理数据集，而不是使用之前缓存的数据。

preprocessing_num_workers: 16：指定了用于数据预处理的工作进程数为 16。

4. 输出设置：

output_dir: saves/DeepSeek-R1-Distill-jiakao-7B/lora/sft：指定了微调后的模型输出路径。

logging_steps: 10：表示每 10 步记录一次训练日志。

save_steps: 500：每 500 步保存一次模型的中间状态。

plot_loss: true：表示绘制训练过程中的损失曲线。

overwrite_output_dir: true：如果输出目录已经存在，将覆盖该目录。

5. 训练参数：

per_device_train_batch_size: 1：每个设备上的训练批次大小为 1。每次只处理一个样本进行训练。

gradient_accumulation_steps: 8：梯度累积的步数为 8。

learning_rate: 1.0e-4：学习率为 0.0001。这是一个合适的超参数，既不会导致收敛过慢，也不会导致无法收敛。

num_train_epochs: 5.0：训练的轮数为 3 轮。一轮是指对整个数据集进行一次完整的遍历。

lr_scheduler_type: cosine：使用余弦退火（cosine annealing）的学习率调度策略。这种策略可以在训练过程中逐渐降低学习率，有助于提高模型的收敛速度和性能。

warmup_ratio: 0.1：热身比例为 0.1。在训练开始时，使用较小的学习率进行热身，然后逐渐增加到指定的学习率。

bf16: true：表示使用 BF16（Brain Floating Point 16）半精度浮点数格式数据类型进行训练。

ddp_timeout: 180000000：分布式数据并行（DDP）的超时时间为 180000000 毫秒（约 50 小时）。

6. 评估设置：

val_size: 0.1：将数据集的 10%作为验证集。在训练过程中，会使用验证集来评估模型的性能，以便及时调整训练策略。

per_device_eval_batch_size: 1：每个设备上的评估批次大小为 1。

eval_strategy: steps：按照指定的步数进行评估。

eval_steps: 500：每 500 步进行一次评估。这与eval_strategy配合使用，确定了评估的频率



## 二 微调结果

**training_eval_loss.png以及training_loss.png**

由图表可见模型loss逐渐变小并趋于稳定，说明训练过程稳定，学习率设置合理，模型有效地学习了训练数据的特征。同时，损失在后期趋于平稳，表明模型已经基本收敛，继续增加训练轮次可能不会带来明显的提升。



## 三 通用能力测试

**通过llama-factory进行测试，各项指标结果在eval文件夹中**

MMLU（Massive Multitask Language Understanding），是一个大规模、多任务的语言理解项目，旨在评估和提升语言模型在各种语言理解任务上的能力。该项目涵盖了广泛的主题和领域，如历史、文学、科学、数学等，通过这些多样化的主题挑战模型的理解能力和知识广度。



## 四 数学能力测试

**通过调用本地部署的大模型端口进行评估，然后调用qwen-max模型进行判断**

测评思路：由于llamafactory 的批量推理不支持 vllm 速度很慢。故把大模型部署成 API 服务，使用异步的调用API回答驾考问题测试集。api端口为http://localhost:8000/v1，输入为json文件的instruction部分，返回instruction和模型回答的json文件形式。由于使用了异步的请求，则必须在所有的请求都完成后才能拿到结果。为了避免程序中途崩溃导致前面的请求的数据丢失，故使用 chunk_size 对请求的数据进行切分，每完成一块数据的请求则把该块数据保存到文件中（具体参考api_call.py）。

回答后的json文件通过调用qwen-max进行答案校对(具体参考judge.py)，最后统计正确个数除以题库总数获得准确率对比。



## 五 数据集

项目主要对两个比较著名的驾校网站的数据进行了抓取，分别是驾校宝典和驾校一点通。

由于两个网站的的网站结构十分相似，不同的是驾校一点通进行了相关的反爬处理。

数据集划分：7：2：1分别用于微调，验证，测试。



## 六 RAG构建数据集去重

我们采用了两步走的策略来进行驾考题目去重：

1. 通过process_jiakao_data.py进行相似题目检测和初步去重

1. 通过validate_data_quality.py验证去重效果和数据质量
