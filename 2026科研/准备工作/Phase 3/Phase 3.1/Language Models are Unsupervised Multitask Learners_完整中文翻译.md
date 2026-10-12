---
aliases:
  - 语言模型是无监督多任务学习者
tags:
  - 论文翻译
  - GPT
  - GPT2
source: language-models-GPT2.pdf
updated: 2026-10-10
---

# 语言模型是无监督多任务学习者

**原题：Language Models are Unsupervised Multitask Learners**

**作者：** Alec Radford\*、Jeffrey Wu\*、Rewon Child、David Luan、Dario Amodei\*\*、Ilya Sutskever\*\*  
**机构：** OpenAI，美国加利福尼亚州旧金山  
**通讯作者：** Alec Radford，alec@openai.com  
**作者注：** \* 与 \*\* 分别表示相应作者贡献相同。  
**年份：** 2019

> [!info] 翻译与阅读说明
> 本译文依据同目录的 [[language-models-GPT2.pdf]]，覆盖全部 24 页：摘要、第 1–7 节、致谢、全部参考文献，以及第 8 节附录 A 的图 5 和表 7–17 样例。五幅图保留英文原图，另附中文图注和必要的图中文字说明；表格、脚注及样例内容转为 Markdown。**参考文献保留英文，不翻译。**人名、数据集名、方法名及提示中的关键字符串保留原名，便于核验。原文错误或易混淆处以“译注”说明；模型生成样例中的不实、重复与不连贯内容按原文保留，不当作事实修订。
>
> 本文中的“我们”指论文作者。文末嵌入 [[GPT2论文理解自测12问与参考答案]]，其内容以独立笔记为维护源；初学者学习从 [[GPT2论文总结]] 的连续主线进入，再按各节链接查阅独立专题。

## 阅读目录

- [[#摘要]]
- [[#1 引言]]
- [[#2 方法]]
- [[#3 实验]]
- [[#4 泛化与记忆]]
- [[#5 相关工作]]
- [[#6 讨论]]
- [[#7 结论]]
- [[#致谢]]
- [[#参考文献（保留英文原文）]]
- [[#8 附录 A：生成样例]]
- [[#学习自测：12 个关键问题与参考答案]]

## 摘要

问答、机器翻译、阅读理解和摘要等自然语言处理任务，通常使用任务专属数据集，通过监督学习来解决。我们证明，当语言模型在一个名为 WebText、包含数百万网页的新数据集上训练时，即使没有任何显式监督，它们也开始学会这些任务。以文档和问题作为条件时，语言模型生成的答案在 CoQA 数据集上达到 55 F1；在没有使用超过 127,000 条训练样例的情况下，其表现达到或超过四个基线系统中的三个。语言模型容量对零样本任务迁移的成功至关重要，增大容量会使各项任务的性能以对数线性的方式改善。我们最大的模型 GPT-2 是一个拥有 15 亿参数的 Transformer，它在零样本设置下，在所测试的八个语言建模数据集中的七个上达到了当时的最佳结果，但对 WebText 的拟合仍然不足。模型生成的样例体现了这些改善，并包含连贯的文本段落。这些发现表明，通过任务自然出现的示范来学习如何执行任务，是构建语言处理系统的一条有希望的路径。

> 译注：“八个数据集中的七个”指**语言建模评估**，不表示翻译、摘要、问答等全部任务均达到最佳水平。论文用“零样本”强调不在目标任务上训练或微调参数；部分任务输入包含示例对，按今天常用的提示术语需要进一步区分，见第 3.7、3.8 节。

## 1 引言

通过结合大型数据集、高容量模型和监督学习，机器学习系统如今已经能在其训练所针对的任务上取得出色的期望表现（Krizhevsky et al., 2012；Sutskever et al., 2014；Amodei et al., 2016）。然而，这些系统依然脆弱，对数据分布的细微变化（Recht et al., 2018）和任务定义的变化（Kirkpatrick et al., 2017）十分敏感。与其说当前系统是有能力的通才，不如说它们是狭窄领域的专家。我们希望走向更通用的系统，让它们能够执行许多任务，并最终不再需要为每一项任务手工创建和标注训练数据集。

构建机器学习系统的主流方法，是收集一批展示目标任务正确行为的训练样例，训练系统模仿这些行为，再在独立同分布（IID）的留出样例上测试其表现。这一方法很好地推动了狭窄领域专家的发展。但图像描述模型（Lake et al., 2017）、阅读理解系统（Jia & Liang, 2017）和图像分类器（Alcorn et al., 2018）面对各种可能输入时经常出现不稳定行为，暴露出这一方法的某些不足。

我们怀疑，在单一领域数据集上开展单任务训练的普遍做法，是当前系统泛化不足的主要原因之一。使用现有架构向稳健系统迈进，很可能需要在广泛的领域和任务上训练并衡量性能。最近提出的 GLUE（Wang et al., 2018）和 decaNLP（McCann et al., 2018）等基准，已经开始研究这一问题。

多任务学习（Caruana, 1997）是改善整体性能的一个有希望的框架。然而，NLP 中的多任务训练仍处于起步阶段。近期工作报告的性能提升较为有限（Yogatama et al., 2019），截至当时最有雄心的两项工作，分别只在总计 10 个和 17 个“数据集、目标”配对上训练（McCann et al., 2018；Bowman et al., 2018）。从元学习的角度看，每一个“数据集、目标”配对，都是从数据集与目标的分布中抽出的一个训练样例。当前机器学习系统需要数百到数千个样例，才能归纳出泛化良好的函数。这意味着，现有方法中的多任务训练，可能也需要同样多的有效训练配对，才能兑现其潜力。若继续采用现有技术，通过大量投入硬推到这一目标，数据集创建和目标设计将很难扩展到所需规模。因此，我们有必要探索其他多任务学习设置。

当前在语言任务上表现最好的系统，结合了预训练和有监督微调。这一方法历史悠久，并逐步走向更加灵活的迁移形式：最初，先学习词向量，再把它们作为任务专属架构的输入（Mikolov et al., 2013；Collobert et al., 2011）；随后，迁移循环网络的上下文表示（Dai & Le, 2015；Peters et al., 2018）；近期工作进一步表明，任务专属架构已经不是必需的，迁移多个自注意力块就已足够（Radford et al., 2018；Devlin et al., 2018）。不过，这些方法仍需要监督训练才能完成任务。当监督数据极少或完全没有时，另一条研究路线已经展示出语言模型完成特定任务的潜力，例如常识推理（Schwartz et al., 2017）和情感分析（Radford et al., 2017）。

本文连接这两条研究路线，继续推动更加通用的迁移方法。我们证明，语言模型可以在零样本设置下完成下游任务，无需修改任何参数或架构。通过展示语言模型在零样本设置下完成广泛任务的能力，我们说明这一方法具有潜力。不同任务上的结果有所不同：有些颇有希望，有些具有竞争力，还有些达到了当时的最佳水平。

![[GPT2_图1_原图.png]]

**图 1：** WebText 语言模型在多项 NLP 任务上的零样本性能随模型规模的变化。阅读理解结果来自 CoQA（Reddy et al., 2018），翻译来自 WMT-14 法语到英语任务（Artetxe et al., 2017），摘要来自 CNN 和 Daily Mail（See et al., 2017），问答来自 Natural Questions（Kwiatkowski et al., 2019）。各项结果的详细说明见第 3 节。

**图中文字说明：** 四个面板从左到右分别是阅读理解、翻译、摘要、问答；横轴均为语言模型参数量，依次为 117M、345M、762M、1542M；纵轴分别为 F1、BLEU、ROUGE 平均值和准确率。虚线表示对应任务的对照系统或人类水平。图中的 Human 是人类水平，Seq2Seq 是序列到序列模型，PGNet 是指针生成网络，Denoising 是去噪方法，Backtranslate 是回译，Lede-3 是取文章开头三句，Random-3 是随机取三句，most freq Q-type answer 是按问题类型返回最常见答案。不同面板的指标不能直接相互比较。

## 2 方法

我们的方法以语言建模为核心。语言建模通常被表述为无监督的分布估计：给定一组样例 $(x_1,x_2,\ldots,x_n)$，其中每个样例都是由符号组成的变长序列 $(s_1,s_2,\ldots,s_n)$。由于语言具有自然的顺序，通常将符号的联合概率分解为条件概率的乘积（Jelinek & Mercer, 1980；Bengio et al., 2003）：

$$
p(x)=\prod_{i=1}^{n}p(s_n\mid s_1,\ldots,s_{n-1}).\tag{1，原文}
$$

> 译注：原 PDF 的公式 (1) 在乘积内部重复使用 $s_n$，与乘积下标 $i$ 不一致，属于下标笔误。上式保留原貌；按上下文的概率链式分解，正确理解应为
> $$p(x)=\prod_{i=1}^{n}p(s_i\mid s_1,\ldots,s_{i-1}).$$
> 此外，原段落也用 $n$ 同时表示样例数量与序列长度，具体阅读时应按上下文区分。

这一方法使得对 $p(x)$ 进行采样和估计，以及对形如 $p(s_{n-k},\ldots,s_n\mid s_1,\ldots,s_{n-k-1})$ 的任意条件分布进行采样和估计，都变得可行。近年来，能够计算这些条件概率的模型，其表达能力已经有显著提升，其中包括 Transformer 等自注意力架构（Vaswani et al., 2017）。

学习执行一个任务，可以在概率框架中表述为估计条件分布 $p(\text{output}\mid\text{input})$。由于通用系统应该能够完成许多不同任务，即使输入相同，也应如此，因此它不仅要以输入为条件，还要以待执行的任务为条件。也就是说，它应该建模 $p(\text{output}\mid\text{input},\text{task})$。这一思想在多任务学习和元学习设置中有不同的形式化表达。任务条件通常在架构层面实现，例如 Kaiser et al.（2017）中的任务专属编码器与解码器；或在算法层面实现，例如 MAML 的内层、外层循环优化框架（Finn et al., 2017）。不过，正如 McCann et al.（2018）所示，语言本身提供了一种灵活方式，可以把任务、输入和输出统一表述为符号序列。例如，一个翻译训练样例可以写成“翻译成法语、英文文本、法文文本”这一序列；类似地，一个阅读理解训练样例可以写成“回答问题、文档、问题、答案”。McCann et al.（2018）证明，单个模型 MQAN 可以从这类格式的样例中推断并执行许多不同任务。

原则上，语言建模也能够学习 McCann et al.（2018）中的任务，而不需要显式说明哪些符号是待预测的输出。因为监督目标与无监督目标相同，只是仅在序列的一个子集上计算，所以无监督目标的全局最小值，也对应监督目标的全局最小值。在这一略带理想化色彩的设置中，Sutskever et al.（2015）讨论的、将密度估计作为原则性训练目标的顾虑可以绕开。问题转而变成：在实践中，我们是否能把无监督目标优化到收敛。初步实验确认，足够大的语言模型能够在这种略显玩具化的设置中开展多任务学习，但学习速度远慢于显式监督方法。

> 译注：这里的“全局最小值”讨论限定在作者构造的理想化任务序列设置中，不能理解为有限容量模型只要降低总体语言损失，就保证改善每一个任务。

从上述明确界定的设置，迈向混杂的“自然环境中的语言”，还有很大距离。Weston（2016）在对话语境中提出，需要开发能够直接从自然语言学习的系统，并给出概念验证：通过向前预测教师的输出，在没有奖励信号的情况下学习问答任务。虽然对话是一种有吸引力的方法，但我们担心它限制过多。互联网上存在大量无需交互通信、可以被动获取的信息。我们的推测是：为了更好地预测自然语言序列，具有足够容量的语言模型将开始学习推断并执行这些序列所展示的任务，不论这些序列是通过什么方式获取的。如果语言模型能够做到这一点，那么实际上，它就在进行无监督多任务学习。我们通过分析语言模型在广泛任务上的零样本表现，检验情况是否如此。

### 2.1 训练数据集

以往大多数工作在单一文本领域上训练语言模型，例如新闻文章（Jozefowicz et al., 2016）、维基百科（Merity et al., 2016）或小说（Kiros et al., 2015）。我们的方法促使我们构建尽可能大且多样的数据集，从尽可能广泛的领域和语境中收集任务的自然语言示范。

Common Crawl 这样的网页抓取档案，是一种有希望提供多样、近乎无限文本的来源。虽然其规模比当前的语言建模数据集大许多个数量级，但它们存在严重的数据质量问题。Trinh & Le（2018）在常识推理研究中使用 Common Crawl，并指出其中有大量“内容大部分无法理解”的文档。我们在 Common Crawl 的初步实验中也观察到了类似问题。Trinh & Le（2018）最好的结果来自 Common Crawl 的一个小型子样本，它只包含与目标数据集 Winograd Schema Challenge 最相似的文档。虽然这是提升特定任务性能的务实办法，但我们希望避免提前假定模型将要执行哪些任务。

**表 1：WebText 训练集中自然出现的英译法与法译英示范。**下列段落译成中文，保留其中作为翻译对象的法文片段及其含义。

1. “我不是世界上最聪明的人，但正如法语里所说：`Je ne suis pas un imbecile`［我不是傻瓜］。”
2. 在 8 月 16 日发布、如今已删除的一篇帖子中，若列特选区的保守党候选人 Soheil Eid 用法语写道：“`Mentez mentez, il en restera toujours quelque chose`”，其意思是：“撒谎吧，撒谎吧，总会有些东西留下。”
3. “我讨厌 `perfume` 这个词，”Burr 说，“法语里的 `parfum` 稍微好一些。”
4. 如果在 29 分 55 秒处仔细听，可以听到两个人用法语交谈：“`-Comment on fait pour aller de l’autre côté? -Quel autre côté?`”，意思是：“怎样才能到另一边去？——哪一边？”
5. 如果这听起来有些牵强，可以考虑这个法语问题：`As-tu aller au cinéma?`，即“你去看电影了吗？”，字面翻译则是“你有没有去电影院／剧院？”
6. “`Brevet Sans Garantie Du Gouvernement`”译成英语的意思是：“已获专利，但不受政府担保。”

> 译注：表 1 展示的是网页中原本存在的片段，法语句子的拼写、语法或字面译法可能不规范；本译文不将其改写为规范教学例句。

作为替代，我们创建了一个新的网页抓取数据集，强调文档质量。为此，我们只抓取经过人类筛选或过滤的网页。人工过滤一次完整的网页抓取结果将极其昂贵，因此，我们以社交媒体平台 Reddit 为起点，收集了所有获得至少 3 karma 的外链。这个指标可以被视为一种启发式信号：其他用户认为这个链接有趣、有教育意义，或只是好笑。

最终得到的数据集 WebText，包含这 4,500 万个链接中提取出的文本子集。为了从 HTML 响应中抽取正文，我们组合使用 Dragnet（Peters & Lecocq, 2013）和 Newspaper 内容提取工具。¹ 本文所有结果均使用 WebText 的一个初步版本：不包含 2017 年 12 月之后创建的链接；经过去重和一些启发式清洗后，包含略多于 800 万篇文档，总计 40 GB 文本。我们从 WebText 中删除了全部维基百科文档，因为维基百科是其他数据集的常见来源，训练数据与测试评估任务的重叠可能使分析复杂化。

¹ Newspaper：[项目仓库](https://github.com/codelucas/newspaper)。

### 2.2 输入表示

通用语言模型（LM）应该能够计算任意字符串的概率，也能够生成任意字符串。当前的大规模语言模型包含转为小写、分词和词表外词元等预处理步骤，这些步骤限制了可建模字符串的空间。将 Unicode 字符串作为 UTF-8 字节序列处理，可以优雅地满足这一要求，Gillick et al.（2015）等工作已经展示过这种做法；然而，在 One Billion Word Benchmark 等大型数据集上，当前的字节级语言模型还无法与词级语言模型竞争（Al-Rfou et al., 2018）。我们在 WebText 上训练标准字节级语言模型时，也观察到了类似的性能差距。

字节对编码（Byte Pair Encoding，BPE；Sennrich et al., 2015）是字符级与词级语言建模之间的一种实用折中：对频繁出现的符号序列，有效地采用接近词级的输入；对不常出现的符号序列，则采用接近字符级的输入。尽管名字中包含“字节”，BPE 的参考实现往往作用于 Unicode 码点，而非字节序列。要建模所有 Unicode 字符串，这些实现必须纳入全部 Unicode 符号；即使还没有添加任何多符号词元，基础词表就将超过 130,000。与 BPE 常用的 32,000 至 64,000 个词元相比，这一规模大得难以承受。相较之下，字节级 BPE 只需要一个大小为 256 的基础词表。

然而，直接对字节序列应用 BPE 会产生不理想的合并，因为 BPE 使用基于频率的贪心启发式来建立词元词表。我们观察到，BPE 会收录 `dog` 这样的常见词的许多版本，因为它们可能以 `dog.`、`dog!`、`dog?` 等不同形式出现。这使有限的词表槽位和模型容量分配得不够合理。为避免这一问题，我们阻止任何字节序列的 BPE 合并跨越字符类别边界。不过，我们为空格设置了例外，这显著改善了压缩效率，而只增加了少量将单词拆成多个词表词元的情况。

这一输入表示使我们能够结合词级语言模型在经验上的优势，以及字节级方法的通用性。由于这一方法可以为任意 Unicode 字符串分配概率，所以我们可以在任意数据集上评估语言模型，不受其预处理、分词方式或词表大小的限制。

### 2.3 模型

我们的语言模型采用基于 Transformer（Vaswani et al., 2017）的架构。模型基本遵循 OpenAI GPT（Radford et al., 2018）的细节，并做了少量修改。层归一化（Ba et al., 2016）被移到每个子块的输入端，类似于预激活残差网络（He et al., 2016）；最后一个自注意力块之后，还增加了一次层归一化。我们使用了修改后的初始化方法，以考虑残差路径上的累积会随模型深度变化。在初始化时，将残差层权重乘以 $1/\sqrt{N}$，其中 $N$ 是残差层数量。词表扩展至 50,257。上下文长度也从 512 个词元增加到 1,024 个，并使用更大的批次大小 512。

**表 2：四种模型规模的架构超参数。**其中 $d_{\text{model}}$ 为模型隐藏表示维度；M 表示百万参数。

| 参数量 | 层数 | $d_{\text{model}}$ |
| --- | ---: | ---: |
| 117M | 12 | 768 |
| 345M | 24 | 1024 |
| 762M | 36 | 1280 |
| 1542M | 48 | 1600 |

## 3 实验

我们训练并评测了四个语言模型，其规模近似按对数均匀间隔分布。架构汇总于表 2。最小模型的规模相当于原始 GPT，第二小的模型规模相当于 BERT 中最大的模型（Devlin et al., 2018）。我们将最大的模型称为 GPT-2，其参数量比 GPT 大一个数量级以上。每个模型的学习率都经过人工调节，以在 WebText 的 5% 留出样本上获得最佳困惑度。所有模型对 WebText 仍然拟合不足，截至论文撰写时，随着训练时间增加，留出集困惑度仍在改善。

> 译注：“相当于”是规模层面的比较；GPT-2 与 BERT 的训练目标和注意力方向不同，不能理解为架构完全相同。本文表 2 使用 117M、345M、762M、1542M 这些原始计数。

### 3.1 语言建模

作为迈向零样本任务迁移的第一步，我们想了解：WebText 语言模型在其主要训练任务——语言建模——上，进行零样本领域迁移时表现如何。由于模型在字节层面工作，不需要有损预处理或分词，所以可以在任何语言模型基准上评估。语言建模数据集的结果，通常以一种量来报告：它是每个规范预测单位的平均负对数概率经过缩放或取指数后的值；预测单位通常是字符、字节或单词。我们计算 WebText 语言模型给整个数据集分配的对数概率，再除以规范单位的数量，以评估同样的量。

对于其中许多数据集，WebText 语言模型实际上是在明显偏离训练分布的条件下接受测试：它们必须预测经过强烈标准化的文本、被拆开的标点和缩写等分词痕迹、打乱的句子，甚至字符串 `<UNK>`。这个字符串在 WebText 中极其罕见，在 400 亿字节里仅出现 26 次。表 3 报告了我们的主要结果，使用了可逆的反分词器，尽可能去除这些分词与预处理痕迹。由于反分词器是可逆的，我们仍能计算数据集的对数概率，也可以将其视为一种简单的领域适配形式。采用这些反分词器后，GPT-2 的困惑度改善了 2.5 至 5。

**表 3：多个数据集上的零样本结果。**这些结果均未在对应任务上训练或微调。PTB 和 WikiText-2 的此前结果来自 Gong et al.（2018）；CBT 来自 Bajgar et al.（2016）；LAMBADA 准确率来自 Hoang et al.（2018），困惑度来自 Grave et al.（2016）；其余此前结果来自 Dai et al.（2019）。

| 模型 | LAMBADA PPL | LAMBADA ACC | CBT-CN ACC | CBT-NE ACC | WikiText2 PPL | PTB PPL | enwik8 BPB | text8 BPC | WikiText103 PPL | 1BW PPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 此前 SOTA | 99.8 | 59.23 | 85.7 | 82.3 | 39.14 | 46.54 | 0.99 | 1.08 | 18.3 | 21.8 |
| 117M | 35.13 | 45.99 | 87.65 | 83.4 | 29.41 | 65.85 | 1.16 | 1.17 | 37.50 | 75.20 |
| 345M | 15.60 | 55.48 | 92.35 | 87.1 | 22.76 | 47.33 | 1.01 | 1.06 | 26.37 | 55.72 |
| 762M | 10.87 | 60.12 | 93.45 | 88.0 | 19.93 | 40.31 | 0.97 | 1.02 | 22.05 | 44.575 |
| 1542M | 8.63 | 63.24 | 93.30 | 89.05 | 18.34 | 35.76 | 0.93 | 0.98 | 17.48 | 42.16 |

**指标说明：** PPL 为困惑度，越低越好；ACC 为准确率，表中数值以百分比表示，越高越好；BPB 为每字节比特数，BPC 为每字符比特数，均越低越好。CBT-CN 指普通名词，CBT-NE 指命名实体。LAMBADA 的 ACC 列包含第 3.3 节说明的停用词过滤。

WebText 语言模型能够很好地跨领域、跨数据集迁移，在零样本设置下改善了八个数据集中的七个的最佳结果。在 Penn Treebank 和 WikiText-2 等小型数据集上，提升尤为明显；这些数据集只有 100 万至 200 万个训练词元。在专门衡量长期依赖的数据集上，例如 LAMBADA（Paperno et al., 2016）和 Children’s Book Test（Hill et al., 2015），也观察到了较大提升。我们的模型在 One Billion Word Benchmark（Chelba et al., 2013）上仍明显落后于先前工作。这可能是多个因素共同造成的：它既是规模最大的数据集，又采用了破坏性很强的预处理；1BW 的句子级打乱移除了全部长距离结构。

### 3.2 儿童图书测试

Children’s Book Test（CBT；Hill et al., 2015）用于考察语言模型在不同词类上的表现：命名实体、名词、动词和介词。CBT 不使用困惑度作为评估指标，而是在自动构造的完形填空测试上报告准确率，任务是从十个候选中预测被删去的单词。按照原始论文提出的语言模型方法，我们计算每个候选及其后续句子内容在语言模型下的概率，并选择概率最大的候选。也就是说，后续句子的概率以所选候选为条件。

![[GPT2_图2_原图.png]]

**图 2：** Children’s Book Test 上的表现随模型容量的变化。人类水平来自 Bajgar et al.（2016），而不是原论文中低得多的估计。左图为普通名词，右图为命名实体；纵轴为准确率，横轴为语言模型参数量；绿色虚线为人类水平，灰色虚线为 Bajgar et al.（2016）的结果。

如图 2 所示，随着模型规模增大，性能持续改善，弥合了这一测试上与人类水平之间的大部分差距。数据重叠分析发现，CBT 测试集中的一本书——Rudyard Kipling 的《丛林之书》（The Jungle Book）——出现在 WebText 中。因此，我们报告没有显著重叠的验证集结果。GPT-2 在普通名词上达到 93.3%，在命名实体上达到 89.1%，均为新的最佳结果。我们还使用反分词器，去除 CBT 中 PTB 风格的分词痕迹。

### 3.3 LAMBADA

LAMBADA 数据集（Paperno et al., 2016）测试系统建模文本长距离依赖的能力。任务是预测句子的最后一个词，而要让人类成功预测，至少需要 50 个词元的上下文。GPT-2 将此前最佳困惑度从 99.8（Grave et al., 2016）改善到 8.6，并将语言模型在这一测试上的准确率从 19%（Dehghani et al., 2018）提升到 52.66%。

检查 GPT-2 的错误后，我们发现，大部分预测都可以合理地续写句子，却不适合作为句子的最后一个词。这说明语言模型没有利用“这个词必须是句子末词”这一额外有用约束。添加停用词过滤，作为这一约束的近似，可以把准确率进一步提升至 63.24%，比该任务此前整体最佳结果高约 4%。此前的最佳方法（Hoang et al., 2018）采用了另一种受限预测设置：模型输出只允许为上下文中出现过的词。对于 GPT-2，这一限制有害而非有益，因为 19% 的答案并未出现在上下文中。我们使用了未经预处理的数据集版本。

> 译注：52.66% 是最大模型未经停用词过滤的准确率，63.24% 是过滤后的结果；表 3 的 45.99% 则属于 117M 小模型。63.24 与此前整体最佳 59.23 相差 4.01 **个百分点**，原文写作“4%”。“此前语言模型的 19%”和“此前整体最佳 59.23%”是不同对照。

### 3.4 Winograd 模式挑战

Winograd Schema Challenge（Levesque et al., 2012）通过衡量系统消解文本歧义的能力，来测试常识推理。近期，Trinh & Le（2018）通过选择语言模型赋予更高概率的歧义消解结果，在这一挑战上取得了显著进展。我们沿用他们的问题形式，并在图 3 中展示模型采用完整评分与部分评分两种技术时的表现。GPT-2 将此前最佳准确率提高了 7%，达到 70.70%。这一数据集很小，只有 273 个样例，因此，我们建议阅读 Trichelair et al.（2018），以更恰当地理解该结果。

![[GPT2_图3_原图.png]]

**图 3：** Winograd Schema Challenge 上的表现随模型容量的变化。横轴为参数量，纵轴为准确率；Full Scoring 为完整评分，Partial Scoring 为部分评分；虚线为此前最佳结果。

> 译注：原文的“提高 7%”按与前作准确率的比较，宜理解为约 7 个百分点。正文没有在这里完整推导两种评分公式，而是指向 Trinh & Le（2018）；不能把本测试的结果扩展为“已经掌握所有常识推理”。

### 3.5 阅读理解

对话问答数据集 CoQA（Reddy et al., 2018）由来自七个不同领域的文档组成，每篇文档配有提问者与回答者围绕该文档进行的自然语言对话。CoQA 测试阅读理解能力，也测试模型回答依赖对话历史的问题的能力，例如“为什么？”

以文档、相关对话历史以及最后一个词元 `A:` 为条件，从 GPT-2 进行贪心解码，在开发集上达到 55 F1。这与四个基线系统中的三个相当或更好，而没有使用那些基线训练所依赖的超过 127,000 条人工收集问答对。当时有监督的最佳系统是一个基于 BERT 的系统（Devlin et al., 2018），其表现正在接近人类的 89 F1。

对于一个没有经过任何任务监督训练的系统来说，GPT-2 的表现令人振奋；不过，检查它的答案与错误后，我们发现，GPT-2 经常使用简单的检索式启发规则，例如遇到“谁”类问题，就从文档中找一个名字作为回答。

### 3.6 摘要

我们在 CNN 和 Daily Mail 数据集（Nallapati et al., 2016）上测试 GPT-2 的摘要能力。为了诱发摘要行为，我们在文章后追加 `TL;DR:`，并用 Top-k 随机采样（Fan et al., 2018）生成 100 个词元，其中 $k=2$。与贪心解码相比，这样可以减少重复，鼓励生成更具有抽象概括性质的摘要。我们将这 100 个词元中的前三个生成句子作为摘要。

虽然从定性上看，这些生成文本像是摘要，如表 14 所示，但它们经常只关注文章靠后的内容，或混淆具体细节，例如车祸涉及多少辆车，以及标志是在帽子上还是衬衫上。在常用的 ROUGE-1、ROUGE-2、ROUGE-L 指标上，生成摘要的表现只是开始接近传统神经网络基线，整体上也仅勉强超过从文章中随机选取三句的方法。去除任务提示后，GPT-2 的综合指标下降了 6.4 分，这说明可以通过自然语言在语言模型中唤起任务专属行为。

**表 4：CNN 和 Daily Mail 数据集上的摘要性能，使用 ROUGE F1 指标衡量。**Bottom-Up Sum 是 Gehrmann et al.（2018）的最佳模型。

| 方法 | R-1 | R-2 | R-L | R-AVG |
| --- | ---: | ---: | ---: | ---: |
| Bottom-Up Sum | 41.22 | 18.68 | 38.34 | 32.75 |
| Lede-3 | 40.38 | 17.66 | 36.62 | 31.55 |
| Seq2Seq + Attn | 31.33 | 11.81 | 28.83 | 23.99 |
| GPT-2，提示 `TL;DR:` | 29.34 | 8.27 | 26.58 | 21.40 |
| Random-3 | 28.78 | 8.63 | 25.52 | 20.98 |
| GPT-2，无提示 | 21.58 | 4.03 | 19.47 | 15.03 |

> 译注：R-AVG 是表中三项 ROUGE 的平均值；21.40 与 15.03 相差 6.37 分，原文四舍五入为 6.4。GPT-2 在 R-2 上略低于 Random-3，不能说每一项摘要指标都超过随机基线。`TL;DR` 是“太长，没读”的常见网络缩写，常被用于引出简短摘要。

### 3.7 翻译

我们测试 GPT-2 是否已经开始学会从一种语言翻译到另一种语言。为帮助它推断所需任务，我们给语言模型提供由 `english sentence = french sentence` 格式的示例对构成的上下文；随后，在最后一个提示 `english sentence =` 后，使用贪心解码生成文本，并将第一个生成句子作为译文。

在 WMT-14 英语到法语测试集上，GPT-2 得到 5 BLEU，略低于使用双语词典逐词替换的方法；该词典来自此前的无监督词翻译工作（Conneau et al., 2017b）。在 WMT-14 法语到英语测试集上，GPT-2 能够利用其很强的英语语言模型，表现明显更好，达到 11.5 BLEU。这超过了 Artetxe et al.（2017）和 Lample et al.（2017）中的若干无监督机器翻译基线，但仍远低于当时最好的无监督机器翻译方法的 33.5 BLEU（Artetxe et al., 2019）。

这一任务上的表现出乎我们的意料，因为在过滤 WebText 时，我们特意删除了非英语网页。为验证这一点，我们对 WebText 运行了一个字节级语言检测器，² 只检测出约 10 MB 的法语数据；这一规模比以往无监督机器翻译研究中常用的单语法语语料小约 500 倍。

² 语言检测器：[CLD2 项目](https://github.com/CLD2Owners/cld2)。

> 译注：输入中包含翻译示例对，所以按今天常见的提示分类，这里属于带上下文示例的迁移，通常称为 few-shot prompting；本文仍按“不进行目标任务参数训练或微调”的口径纳入零样本讨论。保留论文术语时，需要同时保留这一实验条件。删除非英语网页也不等于英语网页中完全没有法语或双语片段，表 1 正展示了此类片段。

### 3.8 问答

测试语言模型包含哪些信息的一种可能方法，是评估它回答事实型问题时有多经常给出正确答案。以往展示这种行为的神经系统，例如所有信息都存储在参数中的 A Neural Conversational Model（Vinyals & Le, 2015），由于缺乏高质量评估数据集，只报告了定性结果。最近提出的 Natural Questions 数据集（Kwiatkowski et al., 2019）是一个有希望用于更定量测试的资源。

与翻译类似，语言模型的上下文先放入一些问答示例对，以帮助模型推断该数据集要求的简短回答风格。使用 SQuAD 等阅读理解数据集中常用的精确匹配指标评估，GPT-2 正确回答了 4.1% 的问题。³ 作为对照，最小模型的准确率没有超过一个极其简单的基线的 1.0%：该基线针对每种问题类型——谁、什么、哪里等——返回最常见答案。GPT-2 正确回答的问题数量是最小模型的 5.3 倍，这说明模型容量可能是迄今神经系统在此类任务上表现不佳的重要因素。

GPT-2 为其生成答案分配的概率校准得较好：在它最有信心的 1% 问题上，准确率为 63.1%。表 5 展示了 GPT-2 在开发集问题上生成的、置信度最高的 30 个答案。不过，GPT-2 的表现仍然远远低于开放域问答系统的 30% 至 50%；后者将信息检索与抽取式文档问答结合使用（Alberti et al., 2019）。

³ Alec 曾认为自己很擅长冷知识；在与 GPT-2 相同的设置下，他对随机抽取的 100 个样例进行了测试，答对了 17 个。原脚注随后打趣道：其实只答对了 14 个，但另外 3 个本来“应该答对”。

**表 5：GPT-2 在 Natural Questions 开发集上生成的、最有信心的 30 个答案，按 GPT-2 分配的概率排序。**根据第 4 节描述的检查过程，这些问题均未出现在 WebText 中。勾叉按原论文保留；问题与答案属于论文发表时的评估样例，不代表现在的事实。

| 问题 | 生成答案 | 原文判定正确 | 概率 |
| --- | --- | :---: | ---: |
| 谁写了《物种起源》？ | 查尔斯·达尔文（Charles Darwin） | ✓ | 83.4% |
| Ubuntu 项目的创始人是谁？ | 马克·沙特尔沃思（Mark Shuttleworth） | ✓ | 82.0% |
| 绿湾包装工队的四分卫是谁？ | 阿隆·罗杰斯（Aaron Rodgers） | ✓ | 81.1% |
| 熊猫是哪个国家的国宝动物？ | 中国 | ✓ | 76.8% |
| 谁提出了相对论？ | 阿尔伯特·爱因斯坦 | ✓ | 76.4% |
| 第一部《星球大战》电影何时上映？ | 1977 年 | ✓ | 71.4% |
| 瑞典最常见的血型是什么？ | A 型 | ✗ | 70.6% |
| 谁被认为是精神分析学的创始人？ | 西格蒙德·弗洛伊德 | ✓ | 69.3% |
| 1969 年谁第一个踏上月球？ | 尼尔·阿姆斯特朗 | ✓ | 66.8% |
| 英国最大的连锁超市是哪家？ | Tesco | ✓ | 65.3% |
| shalom 在英语中是什么意思？ | 和平 | ✓ | 64.0% |
| 《孙子兵法》的作者是谁？ | 孙子 | ✓ | 59.6% |
| 美国陆地面积最大的州？ | 加利福尼亚州 | ✗ | 59.2% |
| 绿藻是哪种生殖方式的例子？ | 孤雌生殖 | ✗ | 56.5% |
| 哪个国家正式使用维克拉姆历？ | 印度 | ✓ | 55.6% |
| 谁主要负责起草《独立宣言》？ | 托马斯·杰斐逊 | ✓ | 53.3% |
| 美国哪个州构成蒙大拿州的西部边界？ | 蒙大拿州 | ✗ | 52.3% |
| 谁在《权力的游戏》中扮演戴佛斯爵士？ | 彼特·丁拉基（Peter Dinklage） | ✗ | 52.1% |
| 谁任命联邦储备系统的主席？ | 珍妮特·耶伦（Janet Yellen） | ✗ | 51.5% |
| 将一个细胞核分成两个遗传信息相同的细胞核的过程叫什么？ | 有丝分裂 | ✓ | 50.7% |
| 谁获得了最多次 NBA 最有价值球员奖？ | 迈克尔·乔丹 | ✗ | 50.2% |
| 哪条河与罗马这座城市有关？ | 台伯河 | ✓ | 48.6% |
| 第一位遭弹劾的总统是谁？ | 安德鲁·约翰逊 | ✓ | 48.3% |
| 2017 年美国国土安全部的负责人是谁？ | 约翰·凯利 | ✓ | 47.0% |
| 欧盟通用货币的名称是什么？ | 欧元 | ✓ | 46.8% |
| 《星球大战》中皇帝叫什么名字？ | 帕尔帕廷 | ✓ | 46.5% |
| 在射击场射击必须持有枪支许可证吗？ | 不需要 | ✓ | 46.4% |
| 谁在 1859 年提出以进化作为生物发展的基础？ | 查尔斯·达尔文 | ✓ | 45.7% |
| 在俄罗斯爆炸的核电站？ | 切尔诺贝利 | ✓ | 45.7% |
| 谁在最初的《终结者》中扮演约翰·康纳？ | 阿诺德·施瓦辛格 | ✗ | 45.2% |

> 译注：表格保留原问题和模型回答中的事实错误、问题措辞问题及原文判定，不进行事实更正。“最有信心的 1% 上达到 63.1%”是选择性评估，不等于所有问题上的准确率；完整开发集的精确匹配率仍为 4.1%。

## 4 泛化与记忆

近期计算机视觉研究表明，常见图像数据集中包含不可忽略数量的近重复图像。例如，CIFAR-10 的训练图像与测试图像有 3.3% 重叠（Barz & Denzler, 2019）。这会导致机器学习系统的泛化性能被高估。随着数据集规模增大，这个问题也越来越可能出现，说明 WebText 中可能存在类似现象。因此，分析有多少测试数据也出现在训练数据中十分重要。

为研究这一问题，我们构建了包含 WebText 训练集词元 8-gram 的 Bloom 过滤器。为提高召回率，我们把字符串规范化为只含小写字母和数字的单词，单词间以一个空格分隔。Bloom 过滤器的构造使误报率上界为 $1/10^8$。我们还生成了 100 万个字符串进一步检查其低误报率，过滤器没有找到其中任何一个。

这些 Bloom 过滤器让我们能够计算：给定数据集中，有多大比例的 8-gram 也出现在 WebText 训练集中。表 6 展示了常见语言模型基准测试集的重叠分析。常见语言建模数据集的测试集与 WebText 训练集有约 1% 至 6% 重叠，平均为 3.2%。有些令人意外的是，许多数据集与其自身训练划分的重叠更大，平均为 5.9%。

**表 6：测试集 8-gram 与训练集重叠的百分比。**“数据集自身训练集”表示基准测试集与该基准训练集的重叠；“WebText 训练集”表示同一个基准测试集与 WebText 的重叠。

| 比较的训练集 | PTB | WikiText-2 | enwik8 | text8 | WikiText-103 | 1BW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 数据集自身训练集 | 2.67% | 0.66% | 7.50% | 2.34% | 9.09% | 13.19% |
| WebText 训练集 | 0.88% | 1.63% | 6.31% | 3.94% | 2.42% | 3.75% |

我们的方法优先保证召回率。人工检查重叠内容发现，虽然其中有许多常见短语，但也有不少较长的匹配，来源于重复数据。这不是 WebText 独有的问题。例如，我们发现 WikiText-103 测试集中的一篇文章，也出现在其训练数据集中。测试集只有 60 篇文章，因此至少存在 1.6% 的重叠。⁴ 按照我们的检查过程，1BW 与其自身训练集的重叠接近 13.2%，这可能更值得担忧。

⁴ 额外重叠中，有相当一部分来自编辑者在主题相关的多篇文章中重复使用某些段落，例如关于朝鲜战争不同战役的文章。

在 Winograd Schema Challenge 中，我们只发现 10 个模式与 WebText 训练集存在任何 8-gram 重叠。其中两个是无关匹配。剩下八个中，只有一个模式出现在会泄露答案的语境里。

在 CoQA 中，新闻领域约 15% 的文档已经出现在 WebText 中，模型在这些文档上的表现约高 3 F1。CoQA 的开发集指标报告五个不同领域上的平均表现；我们测得，跨这些领域的重叠带来约 0.5 至 1.0 F1 的增益。不过，实际训练问题与答案并未出现在 WebText 中，因为 CoQA 的发布时间晚于 WebText 链接的截止日期。

LAMBADA 的平均重叠为 1.2%。对于重叠超过 15% 的样例，GPT-2 的困惑度大约好 2。排除所有存在任何重叠的样例后重新计算指标，困惑度从 8.6 变成 8.7，准确率从 63.2% 降至 62.9%。整体结果变化很小，很可能是因为只有约每 200 个样例中的一个存在显著重叠。

总体而言，我们的分析表明，WebText 训练数据与特定评估数据集的重叠，给报告结果带来了虽小但一致的收益。然而，对于大多数数据集，我们没有观察到比标准训练集与测试集之间原有重叠显著更大的重叠，表 6 展示了这一点。

理解并量化高度相似文本如何影响性能，是一个重要研究问题。更好的去重技术，例如可扩展的模糊匹配，也有助于更好地回答这些问题。目前，我们建议，在为新的 NLP 数据集创建训练／测试划分时，使用基于 n-gram 重叠的去重，作为重要的验证步骤与合理性检查。

检查 WebText 语言模型在其自身留出集上的表现，是另一种可能帮助判断其性能是否归因于记忆的方法。如图 4 所示，WebText 训练集与测试集上的性能相近，并随模型规模增大共同改善。这说明，在许多方面，即使 GPT-2 对 WebText 也仍然拟合不足。

GPT-2 还能写出发现会说话的独角兽的新闻文章，表 13 提供了一个例子。

![[GPT2_图4_原图.png]]

**图 4：** 在 WebText 上训练的语言模型，性能随模型规模的变化。横轴为参数量，纵轴为困惑度；WebText train 是训练集，WebText test 是留出测试集。

> 译注：重叠率是规范化 **8-gram 的比例**，不是逐条样例泄漏比例。这个检查可以检测部分文本重合，但不能排除短片段、语义改写或其他形式的污染；训练与留出表现相近，也不能证明模型从未记忆过文本。正文的“约 1%–6%”为概括性表述，表 6 精确范围为 0.88%–6.31%。

## 5 相关工作

本研究的很大一部分，是衡量在更大数据集上训练的、更大语言模型的性能。这类似于 Jozefowicz et al.（2016）的工作，他们在 One Billion Word Benchmark 上扩展基于循环神经网络的语言模型。Bajgar et al.（2016）也曾通过从 Project Gutenberg 建立一个大得多的训练数据集，补充标准训练集，从而改善 Children’s Book Test 的结果。Hestness et al.（2017）系统分析了不同深度学习模型的性能，如何随模型容量与数据集规模变化。虽然我们的跨任务实验噪声大得多，但结果表明，对一个训练目标中的子任务也存在类似趋势，并且这种趋势持续到了超过 10 亿参数的范围。

生成模型中有趣的习得功能早已有记载，例如 RNN 语言模型中的某些单元会跟踪行宽、检测引号或注释（Karpathy et al., 2015）。对我们更有启发的，是 Liu et al.（2018）的观察：一个被训练来生成维基百科文章的模型，也学会了在不同语言之间翻译人名。

以往工作也探索了过滤和构建大型网页文本语料的其他方法，例如 iWeb Corpus（Davies, 2018）。

语言任务的预训练方法已有大量研究。除引言中提到的方法外，GloVe（Pennington et al., 2014）将词向量表示学习扩展到整个 Common Crawl。文本深度表示学习的一项有影响力的早期工作是 Skip-thought Vectors（Kiros et al., 2015）。McCann et al.（2017）探索了机器翻译模型所产生表示的用途，Howard & Ruder（2018）改进了 Dai & Le（2015）基于 RNN 的微调方法。Conneau et al.（2017a）研究了自然语言推断模型学到的表示的迁移表现，Subramanian et al.（2018）则探索了大规模多任务训练。

Ramachandran et al.（2016）证明，用预训练语言模型来初始化编码器与解码器，可以改善序列到序列模型。近期工作进一步表明，语言模型预训练在针对困难生成任务微调时也有帮助，包括闲聊对话，以及基于对话的问答系统（Wolf et al., 2019；Dinan et al., 2018）。

## 6 讨论

大量研究致力于学习（Hill et al., 2016）、理解（Levy & Goldberg, 2014）和批判性评估（Wieting & Kiela, 2019）有监督与无监督预训练方法得到的表示。我们的结果表明，无监督任务学习是另一个值得探索的研究方向。这些发现可能有助于解释预训练技术为何广泛成功地用于下游 NLP 任务，因为我们展示了：当规模走得足够远时，其中一种预训练技术开始直接学会执行任务，而不需要有监督适配或修改。

在阅读理解上，GPT-2 在零样本设置下能够与有监督基线竞争。然而，在摘要等其他任务上，虽然从定性上看它确实在执行任务，但按定量指标衡量，其表现还只是初级水平。作为研究结果，这具有启发性；但从实际应用角度看，GPT-2 的零样本表现距离可用还很远。

我们研究了 WebText 语言模型在许多经典 NLP 任务上的零样本表现，但还有许多其他任务可以评估。毫无疑问，在许多实际任务上，GPT-2 的表现仍不比随机更好。即使在我们评估过的常见任务上，例如问答与翻译，语言模型也只有在容量足够大时，才开始超过极其简单的基线。

虽然零样本性能为 GPT-2 在许多任务上的潜在表现建立了基线，但微调后的性能上限仍不明确。在某些任务上，GPT-2 完全生成式、非抽取式的输出，与当时许多问答和阅读理解数据集上处于最佳水平的抽取式指针网络输出（Vinyals et al., 2015）差异很大。鉴于此前微调 GPT 的成功，我们计划研究在 decaNLP 和 GLUE 等基准上进行微调；尤其是，GPT-2 增加的训练数据和容量，是否足以克服 BERT（Devlin et al., 2018）揭示的单向表示低效问题，目前还不清楚。

## 7 结论

当一个大型语言模型在足够大且多样的数据集上训练时，它能够在多个领域和数据集上取得良好表现。GPT-2 在零样本设置下，在所测试的八个语言建模数据集中的七个上达到了最佳水平。模型在零样本设置下所能完成的任务具有多样性，这表明：在足够多样的文本语料上训练、以最大化似然为目标的高容量模型，开始学会执行数量惊人的任务，而不需要显式监督。⁵

⁵ 用于下载和使用小模型的初步代码见 [OpenAI GPT-2 项目仓库](https://github.com/openai/gpt-2)。

## 致谢

感谢所有撰写文本、分享链接，并为 WebText 内容点赞的人。数百万人参与创建了用于训练 GPT-2 的数据。也感谢帮助我们建设训练基础设施的 Google 同事，包括 Zak Stone、JS Riehl、Jonathan Hseu、Russell Power、Youlong Cheng、Noam Shazeer、Solomon Boulos、Michael Banfield、Aman Gupta、Daniel Sohn，以及其他许多人。最后，感谢对论文草稿提出反馈的 Jacob Steinhardt、Sam Bowman、Geoffrey Irving 和 Madison May。

## 参考文献（保留英文原文）

Al-Rfou, R., Choe, D., Constant, N., Guo, M., and Jones, L. Character-level language modeling with deeper self-attention. arXiv preprint arXiv:1808.04444, 2018.

Alberti, C., Lee, K., and Collins, M. A bert baseline for the natural questions. arXiv preprint arXiv:1901.08634, 2019.

Alcorn, M. A., Li, Q., Gong, Z., Wang, C., Mai, L., Ku, W.-S., and Nguyen, A. Strike (with) a pose: Neural networks are easily fooled by strange poses of familiar objects. arXiv preprint arXiv:1811.11553, 2018.

Amodei, D., Ananthanarayanan, S., Anubhai, R., Bai, J., Battenberg, E., Case, C., Casper, J., Catanzaro, B., Cheng, Q., Chen, G., et al. Deep speech 2: End-to-end speech recognition in english and mandarin. In International Conference on Machine Learning, pp. 173–182, 2016.

Artetxe, M., Labaka, G., Agirre, E., and Cho, K. Unsupervised neural machine translation. arXiv preprint arXiv:1710.11041, 2017.

Artetxe, M., Labaka, G., and Agirre, E. An effective approach to unsupervised machine translation. arXiv preprint arXiv:1902.01313, 2019.

Ba, J. L., Kiros, J. R., and Hinton, G. E. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

Bajgar, O., Kadlec, R., and Kleindienst, J. Embracing data abundance: Booktest dataset for reading comprehension. arXiv preprint arXiv:1610.00956, 2016.

Barz, B. and Denzler, J. Do we train on test data? purging cifar of near-duplicates. arXiv preprint arXiv:1902.00423, 2019.

Bengio, Y., Ducharme, R., Vincent, P., and Jauvin, C. A neural probabilistic language model. Journal of machine learning research, 3(Feb):1137–1155, 2003.

Bowman, S. R., Pavlick, E., Grave, E., Van Durme, B., Wang, A., Hula, J., Xia, P., Pappagari, R., McCoy, R. T., Patel, R., et al. Looking for elmo’s friends: Sentence-level pretraining beyond language modeling. arXiv preprint arXiv:1812.10860, 2018.

Caruana, R. Multitask learning. Machine learning, 28(1):41–75, 1997.

Chelba, C., Mikolov, T., Schuster, M., Ge, Q., Brants, T., Koehn, P., and Robinson, T. One billion word benchmark for measuring progress in statistical language modeling. arXiv preprint arXiv:1312.3005, 2013.

Collobert, R., Weston, J., Bottou, L., Karlen, M., Kavukcuoglu, K., and Kuksa, P. Natural language processing (almost) from scratch. Journal of Machine Learning Research, 12(Aug):2493– 2537, 2011.

Conneau, A., Kiela, D., Schwenk, H., Barrault, L., and Bordes, A. Supervised learning of universal sentence representations from natural language inference data. arXiv preprint arXiv:1705.02364, 2017a.

Conneau, A., Lample, G., Ranzato, M., Denoyer, L., and Jégou, H. Word translation without parallel data. arXiv preprint arXiv:1710.04087, 2017b.

Dai, A. M. and Le, Q. V. Semi-supervised sequence learning. In Advances in neural information processing systems, pp. 3079– 3087, 2015.

Dai, Z., Yang, Z., Yang, Y., Cohen, W. W., Carbonell, J., Le, Q. V., and Salakhutdinov, R. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860, 2019.

Davies, M. The 14 billion word iweb corpus. https://corpus.byu.edu/iWeb/, 2018.

Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., and Kaiser, Ł. Universal transformers. arXiv preprint arXiv:1807.03819, 2018.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pretraining of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

Dinan, E., Roller, S., Shuster, K., Fan, A., Auli, M., and Weston, J. Wizard of wikipedia: Knowledge-powered conversational agents. arXiv preprint arXiv:1811.01241, 2018.

Fan, A., Lewis, M., and Dauphin, Y. Hierarchical neural story generation. arXiv preprint arXiv:1805.04833, 2018.

Finn, C., Abbeel, P., and Levine, S. Model-agnostic metalearning for fast adaptation of deep networks. arXiv preprint arXiv:1703.03400, 2017.

Gehrmann, S., Deng, Y., and Rush, A. M. Bottom-up abstractive summarization. arXiv preprint arXiv:1808.10792, 2018.

Gillick, D., Brunk, C., Vinyals, O., and Subramanya, A. Multilingual language processing from bytes. arXiv preprint arXiv:1512.00103, 2015.

Gong, C., He, D., Tan, X., Qin, T., Wang, L., and Liu, T.-Y. Frage: frequency-agnostic word representation. In Advances in Neural Information Processing Systems, pp. 1341–1352, 2018.

Grave, E., Joulin, A., and Usunier, N. Improving neural language models with a continuous cache. arXiv preprint arXiv:1612.04426, 2016.

He, K., Zhang, X., Ren, S., and Sun, J. Identity mappings in deep residual networks. In European conference on computer vision, pp. 630–645. Springer, 2016.

Hestness, J., Narang, S., Ardalani, N., Diamos, G., Jun, H., Kianinejad, H., Patwary, M., Ali, M., Yang, Y., and Zhou, Y. Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409, 2017.

Hill, F., Bordes, A., Chopra, S., and Weston, J. The goldilocks principle: Reading children’s books with explicit memory representations. arXiv preprint arXiv:1511.02301, 2015.

Hill, F., Cho, K., and Korhonen, A. Learning distributed representations of sentences from unlabelled data. arXiv preprint arXiv:1602.03483, 2016.

Hoang, L., Wiseman, S., and Rush, A. M. Entity tracking improves cloze-style reading comprehension. arXiv preprint arXiv:1810.02891, 2018.

Howard, J. and Ruder, S. Universal language model fine-tuning for text classification. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), volume 1, pp. 328–339, 2018.

Jelinek, F. and Mercer, R. L. Interpolated estimation of markov source parameters from sparse data. In Proceedings of the Workshop on Pattern Recognition in Practice, Amsterdam, The Netherlands: North-Holland, May., 1980.

Jia, R. and Liang, P. Adversarial examples for evaluating reading comprehension systems. arXiv preprint arXiv:1707.07328, 2017.

Jozefowicz, R., Vinyals, O., Schuster, M., Shazeer, N., and Wu, Y. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.

Kaiser, L., Gomez, A. N., Shazeer, N., Vaswani, A., Parmar, N., Jones, L., and Uszkoreit, J. One model to learn them all. arXiv preprint arXiv:1706.05137, 2017.

Karpathy, A., Johnson, J., and Fei-Fei, L. Visualizing and understanding recurrent networks. arXiv preprint arXiv:1506.02078, 2015.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska- Barwinska, A., et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, pp. 201611835, 2017.

Kiros, R., Zhu, Y., Salakhutdinov, R. R., Zemel, R., Urtasun, R., Torralba, A., and Fidler, S. Skip-thought vectors. In Advances in neural information processing systems, pp. 3294–3302, 2015.

Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pp. 1097–1105, 2012.

Kwiatkowski, T., Palomaki, J., Rhinehart, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Kelcey, M., Devlin, J., et al. Natural questions: a benchmark for question answering research. 2019.

Lake, B. M., Ullman, T. D., Tenenbaum, J. B., and Gershman, S. J. Building machines that learn and think like people. Behavioral and Brain Sciences, 40, 2017.

Lample, G., Conneau, A., Denoyer, L., and Ranzato, M. Unsupervised machine translation using monolingual corpora only. arXiv preprint arXiv:1711.00043, 2017.

Levesque, H., Davis, E., and Morgenstern, L. The winograd schema challenge. In Thirteenth International Conference on the Principles of Knowledge Representation and Reasoning, 2012.

Levy, O. and Goldberg, Y. Neural word embedding as implicit matrix factorization. In Advances in neural information processing systems, pp. 2177–2185, 2014.

Liu, P. J., Saleh, M., Pot, E., Goodrich, B., Sepassi, R., Kaiser, L., and Shazeer, N. Generating wikipedia by summarizing long sequences. arXiv preprint arXiv:1801.10198, 2018.

McCann, B., Bradbury, J., Xiong, C., and Socher, R. Learned in translation: Contextualized word vectors. In Advances in Neural Information Processing Systems, pp. 6294–6305, 2017.

McCann, B., Keskar, N. S., Xiong, C., and Socher, R. The natural language decathlon: Multitask learning as question answering. arXiv preprint arXiv:1806.08730, 2018.

Merity, S., Xiong, C., Bradbury, J., and Socher, R. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843, 2016.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., and Dean, J. Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems, pp. 3111–3119, 2013.

Nallapati, R., Zhou, B., Gulcehre, C., Xiang, B., et al. Abstractive text summarization using sequence-to-sequence rnns and beyond. arXiv preprint arXiv:1602.06023, 2016.

Paperno, D., Kruszewski, G., Lazaridou, A., Pham, Q. N., Bernardi, R., Pezzelle, S., Baroni, M., Boleda, G., and Fernández, R. The lambada dataset: Word prediction requiring a broad discourse context. arXiv preprint arXiv:1606.06031, 2016.

Pennington, J., Socher, R., and Manning, C. Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pp. 1532–1543, 2014.

Peters, M. E. and Lecocq, D. Content extraction using diverse feature sets. In Proceedings of the 22nd International Conference on World Wide Web, pp. 89–90. ACM, 2013.

Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., and Zettlemoyer, L. Deep contextualized word representations. arXiv preprint arXiv:1802.05365, 2018.

Radford, A., Jozefowicz, R., and Sutskever, I. Learning to generate reviews and discovering sentiment. arXiv preprint arXiv:1704.01444, 2017.

Radford, A., Narasimhan, K., Salimans, T., and Sutskever, I. Improving language understanding by generative pre-training. 2018.

Ramachandran, P., Liu, P. J., and Le, Q. V. Unsupervised pretraining for sequence to sequence learning. arXiv preprint arXiv:1611.02683, 2016.

Recht, B., Roelofs, R., Schmidt, L., and Shankar, V. Do cifar-10 classifiers generalize to cifar-10? arXiv preprint arXiv:1806.00451, 2018.

Reddy, S., Chen, D., and Manning, C. D. Coqa: A conversational question answering challenge. arXiv preprint arXiv:1808.07042, 2018.

Schwartz, R., Sap, M., Konstas, I., Zilles, L., Choi, Y., and Smith, N. A. Story cloze task: Uw nlp system. In Proceedings of the 2nd Workshop on Linking Models of Lexical, Sentential and Discourse-level Semantics, pp. 52–55, 2017.

See, A., Liu, P. J., and Manning, C. D. Get to the point: Summarization with pointer-generator networks. arXiv preprint arXiv:1704.04368, 2017.

Sennrich, R., Haddow, B., and Birch, A. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.

Subramanian, S., Trischler, A., Bengio, Y., and Pal, C. J. Learning general purpose distributed sentence representations via large scale multi-task learning. arXiv preprint arXiv:1804.00079, 2018.

Sutskever, I., Vinyals, O., and Le, Q. V. Sequence to sequence learning with neural networks. In Advances in neural information processing systems, pp. 3104–3112, 2014.

Sutskever, I., Jozefowicz, R., Gregor, K., Rezende, D., Lillicrap, T., and Vinyals, O. Towards principled unsupervised learning. arXiv preprint arXiv:1511.06440, 2015.

Trichelair, P., Emami, A., Cheung, J. C. K., Trischler, A., Suleman, K., and Diaz, F. On the evaluation of common-sense reasoning in natural language understanding. arXiv preprint arXiv:1811.01778, 2018.

Trinh, T. H. and Le, Q. V. A simple method for commonsense reasoning. arXiv preprint arXiv:1806.02847, 2018.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. In Advances in Neural Information Processing Systems, pp. 5998–6008, 2017.

Vinyals, O. and Le, Q. A neural conversational model. arXiv preprint arXiv:1506.05869, 2015.

Vinyals, O., Fortunato, M., and Jaitly, N. Pointer networks. In Advances in Neural Information Processing Systems, pp. 2692– 2700, 2015.

Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., and Bowman, S. R. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018.

Weston, J. E. Dialog-based language learning. In Advances in Neural Information Processing Systems, pp. 829–837, 2016.

Wieting, J. and Kiela, D. No training required: Exploring random encoders for sentence classification. arXiv preprint arXiv:1901.10444, 2019.

Wolf, T., Sanh, V., Chaumond, J., and Delangue, C. Transfertransfo: A transfer learning approach for neural network based conversational agents. arXiv preprint arXiv:1901.08149, 2019.

Yogatama, D., d’Autume, C. d. M., Connor, J., Kocisky, T., Chrzanowski, M., Kong, L., Lazaridou, A., Ling, W., Yu, L., Dyer, C., et al. Learning and evaluating general linguistic intelligence. arXiv preprint arXiv:1901.11373, 2019.

## 8 附录 A：生成样例

> 译注：本附录中的续写、摘要、翻译和模型答案均为原论文展示的生成样例，可能包含事实错误、矛盾、重复、无意义内容和不完整句子；以下忠实翻译原文，不把生成文字当作事实，也不替模型修正答案。原文中因固定词元数而截断的开头、结尾仍保留。表 7–11 的“小模型”指最小的 WebText 语言模型，GPT-2 指最大的模型；其参数规模分别为表 2 所列的 117M 和 1542M。

### 8.1 模型容量

为了补充图 4 中报告的更大语言模型在 WebText 上取得的困惑度改善，表 7–11 并排展示了最小 WebText 语言模型和 GPT-2 对随机选取、此前未见过的 WebText 测试集文章所作的续写。

### 8.2 文本记忆

我们观察到，GPT-2 会记住一些在数据集中反复出现很多次的较长字符串，例如著名的引文或演说。举例来说，当以《葛底斯堡演说》的前一句半作为条件时，GPT-2 的 argmax 解码能够复现这篇演说；它在 WebText 中大约出现了 40 次。即使进行不截断的采样，我们也发现模型会先复制这篇演说一段时间，随后才偏离原文，不过仍保持相似的风格。它通常在 100–200 个词元内开始偏离，一旦偏离，生成内容的多样性就会逐步扩大。

为了量化生成样例中出现精确记忆的频率，我们以 WebText 测试集文章为条件，用 GPT-2 生成样例，并将 GPT-2 生成内容的重合率与真实后续文本的重合率进行比较。下面展示了这一分析的结果：GPT-2 重复训练集文本的频率，低于留出文章的基准频率。

![[GPT2_图5_原图.png]]

**图 5：** WebText 测试集和生成样例中，与 WebText 训练集重合的 8-gram 百分比的累积分布函数（CDF）。生成样例以 WebText 测试集为条件，采用 $k=40$ 的 top-$k$ 截断随机采样。大多数样例的重合率低于 1%，其中超过 30% 的样例完全没有重合；测试集重合率的中位数则为 2.6%。

**图 5 图中文字中文对照：** 8-gram overlap with training set 为“与训练集的 8-gram 重合率”；Empirical CDF 为“经验累积分布函数”；samples 为“生成样例”（蓝线）；real 为“真实文本，即 WebText 测试集”（橙线）。

### 8.3 多样性

表 12 展示了对同一段随机选取的 WebText 测试集上下文所作的多次续写，体现了标准采样设置下续写内容的多样性。

### 8.4 鲁棒性

表 13 展示了前文提到的会说话的独角兽新闻文章。我们发现，模型能够处理分布外上下文，但这些样例的质量通常更低。

### 表 7：服装文章的模型容量对比

**上下文（WebText 测试集）**

**穿搭 1：典型搭配**

这是我买下这双鞋时最先想到的一套搭配。它像是这套 Jake Grantham 穿搭的夏季版本；事实上，我的鞋与他的 Nike Racers 配色很接近！我没有穿厚重的 Harris Tweed 外套和牛仔衬衫，而是穿了一件棉质双排扣外套和和一件亚麻衬衫。这两种面料（采用这些颜色）绝对都是夏季必备，因为它们都能搭配深色和和浅色裤子！正如你所看到的，它们与深色牛仔裤和鞋搭得很好。这是一套有点男装／学院风的穿搭。

总的来说，这是一套非常休闲的搭配，因此我给它配了运动鞋。我不喜欢穿一整套羊毛西装再搭运动鞋（GQ 经常展示这种搭配），但我绝对愿意保持休闲感，比如这件棉质双排扣外套。要想在比较随意的男装搭配中穿好运动鞋，休闲面料是关键。我甚至建议用这双运动鞋搭配一套卡其色斜纹棉布西装或白色亚麻西装。只要记得去掉领带，或者穿 T 恤或 Polo 衫；运动鞋配领带有点过头

**小模型续写（117M 参数）**

。

**穿搭 2：高端**

这套穿搭正处在风暴眼中。它融合了 ASICS 对街头服饰的做法（比如 J. Crew），以及 J. Crew 对稍微更传统一点的休闲单品的做法。再说一次，正如我之前提过的，它是一件低预算单品，而它们可能非常昂贵。如果你想做一个休闲运动鞋制造者，这些肯定会成为问题，这也没关系，但如果你想试试他们穿搭中价格更低的版本，就不要犹豫！

这个版本有一个非常低的低领口（这并不是坏事）、略长的腿，以及一双长腿来让一切保持不错。我偏爱的这些单品的颜色的颜色是蓝灰色（这是我的第一个颜色，因为它们是浅白色），或者黑色（这是我所有设计中最喜欢的颜色）。和往常一样，我从这家店里选的颜色，除了设计之外，和那件事没有任何关系。对于一个高端，我会向所有人推荐这个！

**穿搭 3：高亮**

这是一个“高亮”的造型，而我有机会弄点颜色。我觉得这个颜色是基于

**GPT-2 续写（1542M 参数）**

，对我来说！

**穿搭 2：休闲外套**

我知道有些人可能反对穿着高领毛衣搭配运动鞋，但我想忠于自己，所以我选了一件修身的高领毛衣，颜色更接近我当天最喜欢的颜色。总的来说，它是一件经典的高领毛衣，没有什么太花哨的地方。我确实把鞋换成了这双 ASICS Tiger Killshots，而我希望自己当时没有这么做。鞋本身非常舒服，不过，我发现穿着它们穿这双运动鞋，让我的脚显得稍大了点。我会用它们搭配牛仔裤、斜纹棉布裤或任何别的衣服，但绝不会建议单独穿它们。涉及的变量实在太多了。

**穿搭 3：正装**

这是我穿 ASICS Tiger Killshots 时的最后一套搭配。这个版本的 ASICS 是一个更现代、更流线型的版本。这双鞋有着我能想到的最现代、最简约的设计！用了很多黑色，这些运动鞋看起来非常酷。它们在夏天肯定会非常舒服。

对我来说，用它们搭配西装，是拥有一双好运动鞋最简单的办法。基本上就是这样！如果

**表 7 图注：** 随机选取、此前未见过的上下文（原表上方），以及最小模型（原表左侧）和最大模型（原表右侧）未经挑选的续写。上下文长 768 个词元，展示了约 256 个词元的段落。续写长 256 个词元，并完整展示。生成时采用 $k=40$ 的 top-$k$ 随机采样。

### 表 8：西班牙历史文章的模型容量对比

**上下文（WebText 测试集）**

西班牙一些最壮丽的历史景点可以追溯到穆斯林统治时期，其中包括梅斯基塔清真寺（The Mezquita），它最初作为科尔多瓦大清真寺而建；还有同样位于科尔多瓦的麦地那阿扎哈拉（Medina Azahara），如今已成为遗址，但仍可参观，它最初名为 Madinat al-Zahra，是安达卢斯的宫殿；以及格拉纳达的阿尔罕布拉宫，一座华丽而完整的宫殿。还有两座建于穆斯林统治西班牙时期的犹太会堂至今仍然矗立：托莱多的白色圣玛利亚教堂（Santa María la Blanca），以及位于老城区的科尔多瓦犹太会堂。

**收复失地运动与帝国时代［编辑］**

**小模型续写（117M 参数）**

穆斯林统治西班牙的时代与世界其他地区非常不同，它拥有相当丰富的中世纪历史。虽然西班牙帝国存在于公元前 9 世纪后半叶，但到公元 1150 年，它在一个更加繁荣、多样的状态下，相当缓慢地转变成了一个小型、世俗化的帝国。公元 1050 年，那不勒斯皇帝——至少部分以从该地区西北海岸到西海岸的一系列战争和入侵为特征（例如，沿着从直布罗陀一直延伸到西班牙南部的多瑙河，但同一天很快就被罗马军队摧毁）——获得了那不勒斯皇帝的称号。在过去几十年里，他为恢复和维持帝国作出了巨大贡献，其中第一个向西。他的继承者包括 1148 年从教皇宝座夺取权力的斐迪南，而第一位帝位继承者是瑞典国王阿尔弗雷德二世。帝国又成为欧洲大陆上少数几个独立国家之一，虽然这一时期居住在它领土上的人数不超过 2 亿，但人民有很多话要对皇帝说。从 1720 年起，穆斯林占西班牙人口的一半。

**GPT-2 续写（1542M 参数）**

1092 年和 1095 年的收复失地运动（Reconquista，字面意思是“回到这个国家”）促成了伊比利亚半岛上卡斯蒂利亚与阿拉贡统一王国的建立，其中将存在一个穆斯林王朝。这是欧洲最后一次把穆斯林统治视为“本土”的时候，也是受到基督教煽动并组织起来的冲突在欧洲显露自身的最后一次。

穆斯林摩尔人在西班牙的统治，尤其在非穆斯林中，常被称为“征服格拉纳达”，而西班牙有时被称为“格拉纳达”里的西班牙。穆斯林统治的最后几年之后，是西班牙的收复失地运动：随着斐迪南九世于 1189 年 3 月在普瓦捷战役中取胜，以及萨拉丁于 1187 年征服耶路撒冷，穆斯林被驱逐或被迫离开伊比利亚半岛。

摩尔人向北撤退时，西班牙的收复失地运动时期结束了。然而，摩尔人的统治并没有完全被基督徒的统治取代，直到收复失地运动时代恢复，收复失地运动持续了 1229 年至 1492 年。

西班牙最后一个穆斯林国家格拉纳达被征服了，征服者是〔原文在 the 后截断〕

**表 8 图注：** 随机选取、此前未见过的上下文（原表上方），以及最小模型（原表左侧）和最大模型（原表右侧）未经挑选的续写。上下文长 768 个词元，展示了约 256 个词元的段落。续写长 256 个词元，并完整展示。生成时采用 $k=40$ 的 top-$k$ 随机采样。

### 表 9：《吃豆人》文章的模型容量对比

**上下文（WebText 测试集）**

**Pinky**

粉红色幽灵的 AI 被设计成在“感觉”上与红色幽灵的行为相反。Pinky 实际上试图跑到 Pac-Man（吃豆人）的前面。实现方式是：沿着 Pac-Man 移动的方向，把目标设为其当前位置前方 4 个格子的地方。Pac-Man 向上移动时是一个例外。由于代码中的溢出错误，这一计算还会包含一个向左偏移量，其大小等于预期的向上偏移量。

使用 Pac-Man 的预期位置，使玩家能够利用 Pinky 的导航代码。最常见的情况是 Pac-Man 与 Pinky 相互靠得很近。例如，如果它们迎面而行，一旦距离小于 4 个格子，目标格子就会落在 Pinky 身后。如果它在这种情况下遇到交叉路口，就会试图通过转弯折返，以到达目标格子。

玩家经常会在 Pinky 即将接近交叉路口时快速改变方向，做出一种被称为“假动作”的操作。通过在 Pinky 刚要

**小模型续写（117M 参数）**

来到左边时移动目标格子，他们就会获得“正确”的方向。

如果那个位置实际上比 Pac-Man 原先所在的位置更远，玩家的将不会对幽灵的移动作出反应，而幽灵会改为回到玩家的当前位置。这是因为 Pac-Man 不是靠转弯移动的。

幽灵必须去某种其他位置，才能回到它开始的地方，而这可以通过很多方式实现。它常常只是离开那个位置，例如，如果 Pac-Man 跳下悬崖，它会在消失到太空之前回到原来的地点。这些幽灵路线可能存在相当多的可能性，但最明显的情况是，玩家想避免卷入游戏，而目标简单到只要让 Pac-Man 留在游戏中的时间达到它原先时长的三分之一。

**追逐**

对 Pac-Man 而言，这会类似于 Pac-Man 在一场战斗中如何“追上”。

幽灵可以穿过敌人进行追逐。这在很多方面都很有用，对 Pac-Man 和它的 AI 都是如此。与大多数幽灵游戏不同，这个幽灵只是沿着从

**GPT-2 续写（1542M 参数）**

穿过交叉路口时移动目标格子，Pac-Man 就能躲开幽灵的投射物，并回到安全屋的安全地点。

**Buster**

黄色幽灵的 AI 专门被设计成试图阻止 Pac-Man 离开安全屋。玩家必须按空格键，显露安全屋的门，然后走进门里，在下一个房间中找到藏起来的黄色幽灵。

玩家还必须注意：如果幽灵看到 Pac-Man 已经在那里，它就不会尝试离开安全屋。事实上，除非它们已经在安全屋中，否则幽灵一开始就不会尝试离开。

这两种行为基于这样一个事实：玩家并不总是会试图在 Pac-Man（或 Blinky）之前到达安全屋。

如果某个幽灵（或另一个玩家）已经在安全屋中，并试图在 Pac-Man 或 Blinky 一离开时就出来，玩家也必须小心这个漏洞。这时，幽灵将以与“追逐”模式中类似的方式，尝试离开安全屋。

**Pac-Man**

有些幽灵会跟随 Pac-Man。这些幽灵是

**表 9 图注：** 随机选取、此前未见过的上下文（原表上方），以及最小模型（原表左侧）和最大模型（原表右侧）未经挑选的续写。上下文长 768 个词元，展示了约 256 个词元的段落。续写长 256 个词元，并完整展示。生成时采用 $k=40$ 的 top-$k$ 随机采样。

### 表 10：军人传记文章的模型容量对比

**上下文（WebText 测试集）**

下士 Michael P. Goeldin 来自爱尔兰，1860 年 11 月加入 A 连时是一名没有专门技能的劳工。Goldein 在战争中幸存下来。下士 Patrick O’Neal 也来自爱尔兰，他于 1854 年首次入伍，在俄勒冈州的美国第三炮兵团 L 连服役。他回到东海岸，并于 1860 年加入这个连。O’Neal 一直服役到 1874 年，随后被任命为纽约州 Willets Point 国家公墓的主管。下士 Benjamin Browne 来自纽约州奥兰治县，是一名鞋匠。1862 年 8 月，他加入了新成立的纽约第 124 志愿军团，也是同年 10 月转入 A 连的 61 人之一。1864 年 2 月，该连驻扎在 Brandy Station 时，Browne 再次加入该连服役。1867 年服役期满后，他回到平民生活。

6 月 10 日，工匠兵 William Collins 被晋升为下士，很可能是为了弥补渡过詹姆斯河时战斗指挥方面的空缺。Collins 的服役记录并没有体现他为获得此次晋升所展现的品质，但显然他已经克服了一些严重问题。Collins 出生于纽约州 Sacketts Harbor，1853 年 12 月在 22 岁时加入该连，并于 1858 年 12 月再次入伍。就在 1861 年 4 月战争爆发前的一个月，他“翻过山头”（开小差），直到三年后才被抓获。1864 年 3 月 22 日，他被送回该连，并接受审判

**小模型续写（117M 参数）**

，罪名是军事指控，随后被美国军队开除。他又回到该连，并在内战期间被分配到内部事务处。

作为一名工程师，Collins 还在自己的家乡纽约接受过工程训练。在 Union College of Engineering 完成研究生学习后，他同时在美国陆军任职两年。在那里，他在 Daniel Pfeiffer 上尉手下工作，并于 1870 年 5 月晋升为上校，这比 Pfeiffer 上尉入伍晚了三年。Collins 在第一师完成第一年服役之后，回到英国的家中，随后两年在陆军工兵部队服役；到第二年时，他开始体验自己的服役职责。`¡—endoftext—¿`莫斯科，7 月 5 日（路透社）——驻英国的俄罗斯国家媒体以及政府消息人士和专家表示，莫斯科试图淡化叙利亚反对派的“军事集结”，同时继续支持总统 Donald Trump 在美国与墨西哥边境沿线建墙的计划，并称其为一场“军事政变”。

由美国主导的叙利亚战争老兵兼叙利亚事务特使 Michael Flynn（右）在右侧，比划着手势，与驻美国的叙利亚政治分析人士 Ahmet Koryat 一同，和

**GPT-2 续写（1542M 参数）**

，时间是当年 6 月。他因逃兵罪被判枪决，却仍然在现役中服役。

James J. Dall 上尉于 1829 年革命期间出生在弗吉尼亚州，1836 年与他的父亲一同加入陆军。他在弗吉尼亚第二步兵团服役，直到 1845 年，随后去了西部保留地军团；退伍后，他又加入纽约第 25 志愿军团 I 连（1845 年时该团约有 4500 人）。他的一个兄弟 James，在同一旅中，以相同的身份在他的连里服役。这个团一直待在康涅狄格州，直到 1846 年，而 Dall 上尉在退伍前的不同时间里都在 I 连服役。他是墨西哥战争的老兵，曾在圣安东尼奥的 John J. Sullivan 准将幕僚机构中服役两年。在 1863 年的半岛战役中，他指挥一个 23 人的连，负责炮兵连的 54 门重迫击炮和炮艇 Pinta，一艘炮艇以及另外几艘船。1865 年秋天，Dall 上尉在 I 连服现役。两年后，他被晋升为少尉。

John A. O’ 上尉

**表 10 图注：** 随机选取、此前未见过的上下文（原表上方），以及最小模型（原表左侧）和最大模型（原表右侧）未经挑选的续写。上下文长 768 个词元，展示了约 256 个词元的段落。续写长 256 个词元，并完整展示。生成时采用 $k=40$ 的 top-$k$ 随机采样。

> 译注：上下文中 Goeldin 与 Goldein 的拼写差异，以及小模型续写中的 `¡—endoftext—¿` 字样，均按 PDF 可见原文保留；后者之后的文本突然转为另一篇新闻，正是原样例的表现。

### 表 11：软件配置文章的模型容量对比

**上下文（WebText 测试集）**

**Xmonad、Xresources 与字体**

Xresources 可能让人痛苦不堪（至少对我来说是这样）。所以，这里提供一点帮助：

首先，Xmonad 处理字体的一般方式是通过 XFT。Xft 是 X FreeType Interface 库，它负责在 Xmonad／urxvt 中真正绘制字体（Xresources 为在 X 窗口管理器下启动的东西指定一堆资源。单个程序的设置在 Xresources 中像这样指定：`Urxvt.background:background settins here`）。

不管怎样，在 Xresources 中用以下语法设置字体：

```text
urxvt*font:xft:your font name goes here:size=the size of the font goes here,xft:fall-back font 1 goes here
```

所以，这就是通过 Xresources 为 urxvt 指定字体时最基本的部分。

**修复 Cygwin 中 `/.ssh/` 的权限**

在 Windows 的 Cygwin 中，我发现自己无法将 `/.ssh/` 文件夹的权限设为 0600，而 ssh 要求这样设置才能让你使用密钥。我遇到的症状是，无论我怎么做，它总会修改所有者，而组

**小模型续写（117M 参数）**

被设为 `/.ssh/`，这就是当时的情况。这无法彻底修好。我发现，只要用鼠标输入用户名，再试着把组的权限设为“0600”，就能让它好用得多，而我的系统现在就是这样配置的。

然而，自从 Linux 上的 Cygwin 7 起，我发现对我的系统来说，这样做现在已经“没有必要”了！我还安装了一个叫作 chroot 的软件包。它通过允许你安装／升级到你的系统目录来实现这一点。

所以，从 Windows 8 及之后的版本开始，安装它可能需要 chroot，因此我试了又试，发现先在 Linux 上安装这个软件包再卸载它时，情况还不算太糟。按照我上一篇关于 Cygwin 的文章，这个脚本只管做它自己的工作（sudo 命令会完成剩下的事）；如果你尝试它，你就必须记录这个改动。

如果这些还不够让人困惑，以下就是我尝试过的办法，然后我又直接回到自己的 `/.ssh/` 目录和 chroot 脚本。我发现自己终于快把权限用完了（因为我忘了创建一个从文件系统指向 ssh 的符号链接，而且

**GPT-2 续写（1542M 参数）**

，从而使这些文件变成只读。我试过直接启动 ssh，并使用 passwd 之类的程序，把所有者／组改成 0644 权限，但没有用。它令人烦恼，不过还没有到可能出现的最糟糕程度，只是要折腾权限这件事很烦人。

所以我决定反过来这样做：

对 Cygwin 目录中的每个 `.ssh` 文件夹，将每个文件和目录的权限设为 2644。然后用以下命令启动 ssh：

```text
ssh-add /.ssh/
```

第一次连接时，刚才在 Cygwin 中的 `/.ssh` 文件夹权限会被设为 0600。之后，每次再次连接，权限都会变为 0644。

**在 `/.xsessionrc` 中使用的 Xresources 的 Makefile**

这些是在我的 `.xsessionrc` 中做的：

```text
.xprofile: .xsession-errors: color=""
```

例如：

```text
.xprofile: color=blue
```

由于颜色存储在另一个文件中，`/.xsession-errors` 中的颜色将看起来是青色，而不是蓝色。

这主要是因为，到目前为止，只有颜色被赋给了你的

**表 11 图注：** 随机选取、此前未见过的上下文（原表上方），以及最小模型（原表左侧）和最大模型（原表右侧）未经挑选的续写。上下文长 768 个词元，展示了约 256 个词元的段落。续写长 256 个词元，并完整展示。生成时采用 $k=40$ 的 top-$k$ 随机采样。

> 译注：以上软件配置内容同样是用于比较模型续写表现的原始样例，其中包含错误和无效的做法，不能作为实际操作指南。代码及权限数字按原文保留。

### 表 12：同一上下文的六次不同续写

**上下文（WebText 测试集）**

Aaron 喜欢薄荷巧克力蛋糕，但他要求一定要配迷你巧克力豆，所以我就在各层之间撒了一些。我手边还有几块 Peppermint Jo Jos 饼干，所以我把它们压碎，和一些压碎的蛋白霜饼干一起撒了进去，何乐而不为？这简直是薄荷、巧克力豆和松脆饼干美味的大杂烩。我没有计量每种配料用了多少，但尝过成品之后，我希望自己当时能多放一点。你可以加入任何想加的东西——压碎的拐杖糖、薄荷巧克力脆片等。而且不要怕下手重。口感＝好。

如果你没有 7 英寸的蛋糕模，你可以做出 3 层更薄的 8 英寸蛋糕，使用这个

**GPT-2 续写 1**

配方。我做了较长的版本，然后用这个模具做了一层奶油霜糖衣，把其中三分之一放进冰箱，作为顶部糖衣。只要确保在做蛋糕之前就准备好所有这些糖衣，这样当蛋白放在搅拌碗底部时，你就不用等糖衣了。

你想把蛋白打发到硬性发泡吗？那就把蛋糕盘放在一锅微沸的水上方加热，直到它刚刚有一点温热。用不了超过 3–5 分钟；只需足够让它开始

**GPT-2 续写 2**

配方。如果你这么做，而且知道怎么做那个蛋糕，那么你的人生将再也不会一样。

谈到饼干屑或巧克力豆时，我有两条非常简单的规则。1）当我买原糖时，在它冷却下来、并且上面至少有半茶匙糖之前，我不会使用它。这就是我的规则。2）如果我有一个要求使用某样东西的配方，而我想在一个配方中使用它，我不会在网上用别的东西替代它。就像这样。

我找到了一张非常可爱的卡片，上面有可爱的小猫，位于

**GPT-2 续写 3**

配方，这样很可能会得到大约同样数量的蛋糕。

**巧克力酪乳蛋糕**

打印。大块巧克力蛋糕，裹上酪乳，顶部撒上松脆的巧克力碎，再淋上巧克力酱。简单，但相较于巧克力蛋糕，是个不错的变化！作者：Momma Made。

配料。巧克力酪乳蛋糕所需：

- 1 又 1/2 杯无盐黄油，室温
- 1 又 1/2 杯砂糖
- 3 个大鸡蛋加 1 个蛋黄

巧克力糖衣所需：

- 1/3 杯可可粉
- 1 又 3/4 杯糖粉
- 6 盎司半甜巧克〔原文截断于 `semis`〕

**GPT-2 续写 4**

配方。

**蛋白霜饼干**

- 1 又 3/4 杯黄油，软化
- 1 杯糖
- 1 个蛋黄
- 3 大匙重奶油
- 3 又 1/2 至 4 杯面粉
- 一小撮盐

**Peppermint-Jojo 配料**

- 1 又 1/4 杯糖粉
- 1 杯切碎的碧根果
- 1/2 杯切得很细的薄荷叶
- 1/2 杯切碎的新鲜薄荷（约半张）
- 1 茶匙薄荷提取物
- 1/4 茶匙海盐
- 一小撮海〔原文截断于 `sea`〕

**GPT-2 续写 5**

配方。

我在圣诞节早上的两次失败，并不是由于糟糕的糖衣制作技巧。那又是另一个成功的故事。蛋糕润得非常漂亮，而且，嗯，就是一个漂亮的蛋糕。它里面几乎没有空气或水分，但所有巧克力和薄荷都完美地混合在了一起。

有一件事让我有点畏缩，就是所有那些配料——蛋黄、糖、融化的黄油、巧克力豆、融化的巧克力和薄荷——都必须直接混进蛋糕面糊里，这超级费时。有个朋友建议，如果我把面糊放在另一个碗里，我

**GPT-2 续写 6**

配方，它会让你得到约 6–9 份美味的巧克力蛋糕。为了防止各层粘在一起，我用了自己的 8 英寸圆形挞模，垫上烘焙纸，然后从中切出 4 个 7 英寸的圆形蛋糕，把蛋糕层夹在它们之间（这对单层巧克力蛋糕也同样管用）。准备烤这个蛋糕时，取出烘焙纸，将烤箱预热到 350 度，烘烤约 30–40 分钟，直到表面呈金黄色，而且插入的测试针拔出来时是干净的。

**配 Peppermint Jo 的巧克力酪乳千层蛋糕**

**表 12 图注：** GPT-2 从同一上下文（来自 WebText 测试集）生成的未经挑选的续写。上下文长 384 个词元（展示时有截断），生成内容长 128 个词元。生成时采用 $k=40$ 的 top-$k$ 随机采样。

> 译注：原表的上下文以 “out of this” 结束，六段续写均从 “recipe” 开始，以上分别保留为“使用这个”与“配方”的衔接。第 6 段的 350 度未在原文注明单位，因此不自行补写。配方中的矛盾和异常量词按原文保留。

### 表 13：会说英语的独角兽

**上下文（人工撰写）**

在一项令人震惊的发现中，科学家在安第斯山脉一处偏远、此前从未探索过的山谷里，发现了一群独角兽。更让研究者惊讶的是，这些独角兽说着完美的英语。

**GPT-2**

科学家根据它们独特的角，将这一种群命名为“奥维德独角兽”。这些四只角、银白色的独角兽此前不为科学界所知。

如今，在将近两个世纪之后，是什么引发了这一奇怪现象的谜团终于解开了。

来自拉巴斯大学的进化生物学家 Jorge Pérez 博士与几位同伴正在探索安第斯山脉，这时他们发现了一个小山谷，里面没有其他动物，也没有人类。Pérez 注意到，山谷里似乎有一处天然喷泉，周围环绕着两座岩石山峰和银色的积雪。

随后，Pérez 和其他人进一步深入山谷。“等到我们登上一座山峰的顶部时，水看起来是蓝色的，上面有一些晶体，”Pérez 说。

Pérez 和他的朋友们看到这群独角兽时惊呆了。从空中就能看见这些生物，不必大幅移动就能看到它们——它们近得让人可以摸到它们的角。

在检查这些奇异生物时，科学家们发现，这些生物还会说一些相当普通的英语。Pérez 表示：“例如，我们可以看出，它们有一种共同的‘语言’，类似一种方言，或辩证法。”

Pérez 博士认为，独角兽可能起源于阿根廷；据认为，那里的这些动物是一个已经失落的人类族群的后代，这个族群在其他人类抵达南美洲那些地区之前就生活在那里。

尽管它们的起源仍不清楚，一些人认为，也许这些生物是在一个人类与一只独角兽相遇时产生的，时间是在出现人类文明之前。根据 Pérez 的说法，“在南美洲，这种事件似乎相当常见。”

不过，Pérez 也指出，要确切知道独角兽是否确实是一个失落的外星种族的后代，很可能只有通过 DNA 才能做到。“但它们似乎能够用英语相当顺畅地交流，我认为这是进化的标志，或者至少说明社会组织发生了变化，”这位科学家说。

**表 13 图注：** GPT-2 以分布外上下文为条件进行生成。这一例子是从采用 $k=40$ 生成的 10 个样例中挑选出来的。

### 表 14：CNN 和 Daily Mail 测试集上的摘要样例

#### 样例 1：复制史前洞穴壁画

**文章**

3.6 万年前，史前人类在如今法国境内一处洞穴粗糙的石灰岩墙壁上，画出了种类惊人丰富的史前野兽。

如今，在尖端技术的帮助下，肖维-蓬达尔克洞穴（Chauvet-Pont-d’Arc Cave）中的这些艺术作品被复制下来，建成了世界上最大的仿真洞穴。

这座人工洞穴名为 Caverne du Pont-d’Arc，建在法国南部 Vallon-Pont-D’arc 原址几英里之外，里面包含 1000 幅精心复制的绘画，以及约 450 块骨骼和其他特征……

3.6 万年前，穴居男女在一处洞穴粗糙的石灰岩墙壁上，画出了种类惊人丰富的史前野兽，如今人们已经建成了它的复制品（见图）。

……

**GPT-2 摘要**

法国南部 Vallon-Pont-D’arc 的原址是联合国教科文组织世界遗产，也是已知最古老、保存最完好的人类装饰洞穴。仿真洞穴建在法国南部 Vallon-Pont-D’Arc 原址几英里之外。洞穴中包含 14 种不同动物的图像，包括披毛犀、猛犸象和大型猫科动物。

**参考摘要**

这个洞穴模仿了法国著名的 Caverne du Pont-d’Arc，后者是最古老、保存最完好的人类装饰洞穴。复制品包含全部 1000 幅绘画，其中包括 425 个，如披毛犀和猛犸象。人们使用三维建模和变形技术复制了极其细微的细节，这些技术也常用于拍摄宽银幕图像。这座现代洞穴还包括复制的熊爪印、骨骼，以及原洞穴中保存下来的细节。

#### 样例 2：富豪的私人潜艇

**文章**

“如果你能找到我的潜艇，它就是你的了，”俄罗斯石油亿万富翁 Roman Abramovich 曾经说过。

看来，这位寡头兼切尔西足球俱乐部老板，并不是唯一一个在水下探索上豪掷资金的富有商人；他的潜艇只是价值 3 亿英镑的超级游艇 Eclipse 附带的配件之一（非常适合神不知鬼不觉地上岸）。

海洋被称为地球真正的“最后边疆”，但它们仍然如此缺乏探索，以至于亿万富翁们排着队购买船只，只为一睹世界海洋中壮观的海底景色和令人难以置信的野生生物。

所以，如果你银行里闲放着几百万，而且想拍几张度假照片、发几条足以压过其他所有人的 Instagram 帖子，你也许会想迅速买下一艘这种船……

尽兴的时光：OrcaSub 能把你带到水面以下 2000 英尺，在两个加压的 Perspex 有机玻璃观景舱中，实现最佳的探索体验。

……

**GPT-2 摘要**

U-Boat Worx 潜艇能够下潜至 984 英尺（300 米），也能够潜到海底。它可用于潜水、浮潜，甚至浮潜潜水。这艘潜艇还配备了一个浮潜平台。

**参考摘要**

乘坐最先进的水下海洋载具，体验成为詹姆斯·邦德电影反派的感觉。有些可以从游艇变为潜艇，另一些则可以脱离出来，进行快速探索。Oliver’s Travels 在其潜艇上提供一个 Mile Low 套餐，配有厨师和管家，每晚价格为 17.5 万英镑。

#### 样例 3：逃离也门空袭

**文章**

Amina Ali Qassim 坐着，最小的孙辈坐在她膝上，她用头巾擦去眼泪。

这个才几个月大的女婴，就是空袭开始的那一夜，她拼命试图捂住耳朵的孩子。她说，自己在也门比里姆岛的一座村庄清真寺里彻夜未眠，在婴儿的哭声中数着爆炸声。

但情况本来可能更糟。第一枚导弹落下来时，他们本来可能仍在家里。

“我们的邻居朝我丈夫喊：‘你们必须离开，他们来了。’然后我们就跑了。我们刚离开房子，第一枚导弹就落在旁边，接着第二枚落在了房子上。它把所有东西都烧成了平地，”Qassim 告诉我们。

……

**GPT-2 摘要**

也门正处于内战之中。沙特阿拉伯正在领导联合空袭行动。它已经轰炸也门两个多月了。

**参考摘要**

Amina Ali Qassim 的家人在逃离也门前，在一座清真寺里寻求庇护。数千名像他们一样的人正在登船驶向吉布提。沙特阿拉伯一直在猛烈轰炸也门，试图击败胡塞叛军。

**表 14 图注：** GPT-2 在 CNN 和 Daily Mail 数据集测试集上生成的摘要，以及参考摘要。

> 译注：样例 1 的参考摘要原文写作 “all 1,000 paintings which include 425 such as a woolly rhinoceros and mammoths”，其中 425 的具体所指并不清楚，这里按原文保留，没有推定或补写单位。文章中的省略号和图片提示为原表已有内容；原表没有嵌入新闻图片。

### 表 15：英语与法语之间的翻译样例

> 译注：为便于核对翻译错误，以下保留全部法文原文，再附中文释义。法文模型输出中存在语法错误、重复、错误选词和截断，中文释义尽量反映这些缺陷；其目的不是将错误法文改写成通顺的正确译文。

#### 样例组 1：专辑再版

**英语参考文本（中文译文）**

这次再版名为 *The Next Day Extra*，以三张光盘的形式呈现：原专辑、未发表的录音室录音和混音版本，再加上一张 DVD，包含此前已公开的四支视频。

**GPT-2 法语译文（原文）**

> Les nouvelles re-releases, tout en premier disc, nécessaire de l’album, un studio session et remixes, plus une DVD de l’écran de quelques clips qui ont été déjà échappés.

**中文释义：** 新的再发行，全都在第一张盘，专辑的必要，一段录音室录音和混音，再加上一张屏幕的 DVD，有些已经逃出来的视频片段。

**法语参考文本（原文）**

> Un homme expliquait que le fonctionnement de la hernia fonctionnelle qu’il avait reconnaît avant de faire, le fonctionnement de la hernia fonctionnelle que j’ai réussi, j’ai réussi.

**中文释义：** 一个男人解释说，功能性疝气的运作，他在做之前已经承认，功能性疝气的运作，我成功了，我成功了。

**GPT-2 英语译文（中文译文）**

这次发行，最初以三张光盘发行：原专辑、未发表的录音室录音和混音，一张 DVD，包含前一天发布的四支视频。

> 译注：原 PDF 这一组的“法语参考文本”确实写的是上述疝气相关句子，且与下一组的 GPT-2 法语输出完全相同，与本组专辑主题及右侧英语输出不匹配；已核对原表版面，保留原文，不替换为自行推测的正确参考译文。右侧英语输出中还混有法文 `inédites`。

#### 样例组 2：免费的疝气手术

**英语参考文本（中文译文）**

一名男子解释说，他接受的免费疝气手术将使他能够重新工作。

**GPT-2 法语译文（原文）**

> Un homme expliquait que le fonctionnement de la hernia fonctionnelle qu’il avait reconnaît avant de faire, le fonctionnement de la hernia fonctionnelle que j’ai réussi, j’ai réussi.

**中文释义：** 一个男人解释说，功能性疝气的运作，他在做之前已经承认，功能性疝气的运作，我成功了，我成功了。

**法语参考文本（原文）**

> Un homme a expliqué que l’opération gratuite qu’il avait subie pour soigner une hernie lui permettrait de travailler à nouveau.

**中文译文：** 一名男子解释说，他接受的治疗疝气的免费手术将使他能够重新工作。

**GPT-2 英语译文（中文译文）**

一个男人告诉我，他们曾答应给他的那笔手术小费，将不会让他能够旅行。

#### 样例组 3：Kerry 谈政府的自动运行事务

**英语参考文本（中文译文）**

Kerry 先生通过视频连线，在伦敦一场开放政府会议上发言时说：“毫无疑问，总统、我和政府中的其他人，确实已经了解到一些一直在自动运行的事情，因为这种能力一直存在，可以追溯到第二次世界大战和冷战期间非常艰难的那些年份，然后当然还有 9·11。”

**GPT-2 法语译文（原文）**

> Le premier conference de l’édition internationale en ligne sur la vide, le ministre de l’Édition internationale a été vide, lui dit : ”Il n’y a qu’une question que le président et moi et tous les autres personnes d’une administration qui a été évident par la fois, par la part de la guerre, par la part de la guer

**中文释义：** 第一场关于空的线上国际版会议，国际版部长曾经是空的，他对他说：“只有一个问题，总统和我，以及某个行政机构中的所有其他人，这个机构曾经通过那次、通过战争的部分、通过战〔原文截断于 `guer`〕

**法语参考文本（原文）**

> S’exprimant lors d’une conférence intergouvernementale à Londres par liaison vidéo, M. Kerry a déclaré: ”Il est indéniable que le Président, moi-même et d’autres membres du gouvernement avons pris connaissance de certaines choses en mode pilote automatique parce que nous en avions la possibilité, dès la Seconde guerre mondiale et jusqu’aux années difficiles de la Guerre froide, puis bien sûr le 11 septembre.”

**中文译文：** Kerry 先生通过视频连线，在伦敦一场政府间会议上发言时说：“不可否认，总统、我本人和政府中的其他成员，在自动驾驶模式下得知了某些事情，因为我们有这种可能性，从第二次世界大战开始，一直到冷战的艰难岁月，然后当然还有 9 月 11 日。”

**GPT-2 英语译文（中文译文）**

在美国与伦敦之间的一场会议中，国务卿 John Kerry 说：“不可否认，总统、我本人以及政府中的其他人，已经知道过去作出的某些某些选择，以便能够用更自动化的方式做某些事情。”

> 译注：这一英语输出中的 `conférence` 保留了法文拼写，`indeniable` 也不是规范的英语拼写；重复的 “certain certain” 译为“某些某些”，不予修正。

**表 15 图注：** GPT-2 生成的英译法和法译英译文。

### 表 16：CoQA 对话问答样例——奥运火炬接力

**上下文（文章及此前的问答对）**

2008 年夏季奥运会火炬接力于 2008 年 3 月 24 日至 8 月 8 日举行，时间在 2008 年夏季奥运会之前，主题为“同一个世界，同一个梦想”。火炬接力计划于 2007 年 4 月 26 日在中国北京公布。组织者还将这次接力称为“和谐之旅”；它持续 129 天，将火炬传递了 137000 千米（85000 英里），这是自 1936 年夏季奥运会前开始这一传统以来，历届奥运火炬接力中最长的距离。

3 月 24 日，火炬在奥运会的发源地——希腊奥林匹亚——点燃后，前往雅典的泛雅典体育场，然后来到北京，于 3 月 31 日抵达。从北京出发后，火炬沿着一条途经六大洲的路线前进。火炬曾到访丝绸之路沿线的城市，象征中国与世界其他地区之间的古老联系。接力还包括从中国一侧，将圣火带上位于尼泊尔与中国西藏边境的珠穆朗玛峰峰顶；这一侧为了该活动而特别封闭。

**此前的问答对**

**问：** 主题是什么？  
**答：** “同一个世界，同一个梦想”。

**问：** 这次接力的长度是多少？  
**答：** 137000 千米。

**问：** 比以前的接力更长吗？  
**答：** 不是。

**问：** 接力从哪里开始？  
**答：** 希腊奥林匹亚。

**问：** 那个地方有什么特别之处吗？  
**答：** 奥运会的发源地。

**问：** 他们之后去了哪里？  
**答：** 雅典。

**问：** 接力持续了多少天？  
**答：** 七天。

**问：** 他们去了什么著名地标吗？  
**答：** 泛雅典体育场。

**待回答的问题**

**问：** 那他们爬了什么山吗？  
**答：**

**模型答案：** 珠穆朗玛峰。  
**众包标注者答案：** 不知道；是的；是的；是的。

**表 16 图注：** 选取的 CoQA 续写样例。

> 译注：这里此前的问答对中，“不是”“七天”等回答与文章信息存在矛盾；它们确实出现在原表中，仍按原文保留。最终模型答案指出了山名，而多数人工答案是对“是否爬山”的肯定回答，两者的回答形式不同。

### 表 17：CoQA 对话问答样例——Catherine 与她的狗

**上下文（文章及此前的问答对）**

54 岁的秘书 Catherine Green 去哪里，Tom 都跟着。她上班时，他在她的办公室里走来走去，还和她一起购物。“大多数人似乎并不介意 Tom，”Catherine 说，她觉得他非常好。“他是我的第四个孩子，”她说。她也许把他当作自己的儿子，并这样对待他。他四处活动，买他的食物，付他的医疗账单和税款，但事实上，Tom 是一只狗。

Catherine 和 Tom 住在瑞典；在这个国家，每个人都被要求按照政府制定的规则，过有秩序的生活，政府也为人民提供高水平的照护。这种水平的照护需要花钱。

瑞典人对一切都要纳税，所以他们发现养一只狗意味着更多税款时，并不会惊讶。有些人为获得养狗的权利，每年要支付多达 500 瑞典克朗的税款；政府将这笔钱用于狗医院，有时也用于患病狗的医疗。不过，这类治疗大多很昂贵，所以主人往往决定为他们的狗提供健康，甚至生命。

在瑞典，狗主人必须赔偿狗造成的任何损害。一名瑞典养犬俱乐部官员解释了这意味着什么：如果你的狗跑到路上，被一辆经过的汽车撞到，作为主人，你必须赔偿这辆车受到的任何损害，即使你的狗已经在这次事故中死亡。

**此前的问答对**

**问：** Catherine 多大？  
**答：** 54 岁。

**待回答的问题**

**问：** 她住在哪里？  
**答：**

**模型答案：** 斯德哥尔摩。  
**众包标注者答案：** 瑞典；瑞典；在瑞典；瑞典。

**表 17 图注：** 选取的 CoQA 续写样例。

> 译注：文章只支持“瑞典”，未提供具体城市；模型回答“斯德哥尔摩”超出了文章证据。上下文中的 “offer health and even life for their dog” 语义可疑，可能存在漏字，但这里按可见原文直译，不补成自行推断的保险内容。

## 学习自测：12 个关键问题与参考答案

下面嵌入独立维护的自测笔记。在 Obsidian 中可以先作答，再展开每题参考答案；修改答案时只需更新原笔记，此处会同步显示。

![[GPT2论文理解自测12问与参考答案]]

## 配套初学者详解

[[GPT2论文总结]] 按“问题 → 概率目标 → 自然任务示范 → 数据与实现 → 任务评估 → 证据边界”的顺序提供连续解释、GPT1 对照与复现清单。可独立学习的定义、教学例子、公式、指标与实验细节已移入各节链接的九篇专题，总文与专题共同组成详细解读。
