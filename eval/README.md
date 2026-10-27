# PDF 翻译的评测

这里的脚本用来衡量 PDF 翻译前的分析结果：

- 分段：段落边界切得对不对。
- 行内公式：正文里的公式有没有整体做成占位符，不被送去翻译。
- 两个模型（公式检测、版面分析）的框：换模型、量化前后框有多少变化。

分段和行内公式的标准答案取自 arXiv 论文的 LaTeX 源码（评测集，36 篇）。另有 15 份没有源码的 PDF 作回归语料，用来列出两个版本之间输出的差别，由人判断好坏。

## 目录

| 路径 | 内容 |
|---|---|
| `env.sh` | 各脚本共用的路径设置 |
| `papers.tsv` | 评测集的论文列表：arXiv 编号（带版本号）、分类、页数、PDF 的 SHA-256、标题 |
| `tools/` | 评测集：下载论文、生成标准答案、出计划、打分 |
| `regress/` | 回归语料：出计划，比较两个版本；`files.tsv` 是文件列表 |
| `models/` | 模型实验：int8 量化、换公式检测模型、测速 |

下文的「计划」指程序对一篇 PDF 切出的翻译单元列表，每个单元记录页码、是否跳过（保留原文）和送去翻译的文本。出计划只做翻译前的分析，不调用翻译服务。每次出计划时给一个标签，计划存为 `<编号>.<标签>.json`，两个标签的计划可以相互比较。

## 准备

### 环境

- 项目的 `.venv`，装好 `requirements.txt` 和 onnxruntime。生成 int8 模型还要装 onnx：`.venv/bin/pip install onnx`。
- 两个模型文件在程序的「版面模型」页面下载，默认保存在 `data/models/`：
  - 公式检测模型 `pix2text-mfd-1.5.onnx`
  - 版面模型 `doclayout_yolo_docstructbench_imgsz1024.onnx`
- 脚本用 bash 运行，例如 `bash eval/tools/run.sh . rules`。

shell 脚本从 `env.sh` 读取路径，以下环境变量可以覆盖默认值：

| 变量 | 含义 | 默认值 |
|---|---|---|
| `EVALSET` | 评测数据目录，存放论文、标准答案、计划、框 | `~/doc-translator-evalset` |
| `PY` | 带项目依赖的 Python | 项目目录下的 `.venv/bin/python` |
| `CODE` | 出计划、出框所用的代码目录，可以指向某个提交的副本 | 项目目录 |
| `REGRESS_PDFS` | 回归语料的 PDF 所在目录 | 项目目录下的 `data/jobs` |
| `WORK` | 实验产物（int8 模型等）的目录 | `$EVALSET/work` |

评测数据目录的结构：

| 子目录 | 内容 |
|---|---|
| `raw/<编号>/` | 论文：`paper.pdf`、LaTeX 源码 `src/`、`meta.txt` |
| `gold/`、`mathgold/` | 分段、行内公式的标准答案 |
| `mfd/` | 公式检测模型的框，`<编号>.<标签>.json` |
| `regions_<名字>/` | 版面模型的框，每篇一个 `<编号>.json` |
| `plans/`、`mplans/` | 分段计划、带公式文字的计划 |
| `results/` | 分段打分的逐篇结果 |
| `regress/` | 回归语料的框（`mfd/`、`regions_<名字>/`）和计划（`plans/`） |

### 下载论文

```bash
mkdir -p $EVALSET/raw
.venv/bin/python -I eval/tools/fetch.py eval/papers.tsv $EVALSET/raw
```

脚本按列表里的版本号下载 PDF 和 LaTeX 源码，源码只解出 `.tex`、`.bbl`、`.sty`、`.cls`。arXiv 要求请求间隔 3 秒以上，36 篇约需 4 分钟。下载的文件来自外部，评测脚本只把源码当文本读，不编译；运行 Python 时都加 `-I`，不从当前目录和脚本所在目录加载模块。下载的 PDF 和列表里的 SHA-256 不同时，脚本会报出来。

### 生成标准答案

```bash
mkdir -p $EVALSET/gold $EVALSET/mathgold
for d in $EVALSET/raw/*/; do id=$(basename $d)
  .venv/bin/python -I eval/tools/gold.py $d $EVALSET/gold/$id.json
  .venv/bin/python -I eval/tools/mathgold.py $d $EVALSET/mathgold/$id.json
done
```

两个脚本的输出只取决于源码，同一份源码每次生成的结果相同。

### 回归语料

`regress/files.tsv` 列出 15 份文件的名称、页数、SHA-256 和来源（见文末）。把这些 PDF 放进一个目录，用 `REGRESS_PDFS` 指过去。在线下载到的版本可能和列表里的不同，以 SHA-256 为准。

## 标准答案和打分规则

### 分段

`gold.py` 从源码里取段落：段落在空行、`\par`、章节命令、块级环境、`\item` 处断开；段内的行间公式两边算「不计」的边界，拆开和不拆开都不算错。36 篇共 2073 段、1689 处段落边界。

`score.py` 在计划里找每段的开头和结尾（段首、段尾 12 个词以内、全文只出现一次的 4 词片段），统计两类错误：

- 拆错：一段的开头和结尾落在不同单元。
- 并错：相邻两段的前段结尾和后段开头落在同一个单元。

标准答案不覆盖参考文献，参考文献的分段要用 `joindiff.py` 列出来人工检查。

### 行内公式

`mathgold.py` 记下源码里每个 `$…$`、`\(…\)` 公式和两边各 4 个正文词（锚点）。`mathscore.py` 按锚点在计划里找到公式所在的位置，把每处公式记为：

| 记号 | 含义 |
|---|---|
| `full` | 公式全部在占位符里 |
| `minor` | 占位符外只漏了数字、单位这类整词（如 50 Myr），不影响翻译 |
| `bad_partial` | 占位符外漏了变量、希腊字母或数学符号 |
| `bad_missed` | 没有做成占位符 |
| `empty` | 两个锚点之间没有可比较的文字 |
| `lost` | 锚点没找到，不计分 |

正确率 = (`full` + `minor` + `empty`) / 计分的总数。锚点词被做成占位符的记为误判（输出里的 `fp`）。结果按正文字体分成两组：Computer Modern 字体排的（`cm`）和其他字体排的（`other`，如 Times），后一组的斜体变量和正文斜体是同一种字体，更难区分。

## 评测流程

以下命令都在项目目录下运行，并且已设好数据目录：

```bash
export EVALSET=~/doc-translator-evalset WORK=~/doc-translator-evalset/work
```

命令里的第一个参数 `.` 是出计划所用的代码目录，换成某个提交的副本就能比较两个版本。

### 只用规则

```bash
bash eval/tools/run.sh . rules
```

出分段计划并打分，逐篇结果写到 `results/rules.txt`，最后一行是总数。行内公式用 `mathrun.sh`：

```bash
bash eval/tools/mathrun.sh . rules
```

### 带模型

模型推理慢，先把框存下来，改了规则以后只读框重新出计划。

1. 用公式检测模型给 36 篇出框并打分，框存为 `mfd/<编号>.rect.json`：

   ```bash
   bash eval/models/run_formula_model.sh data/models/pix2text-mfd-1.5.onnx rect:1024
   ```

2. 用版面模型出框，存到 `$EVALSET/regions_rect/`，再和第 1 步的公式框一起打分段的分：

   ```bash
   bash eval/models/run_layout_model.sh data/models/doclayout_yolo_docstructbench_imgsz1024.onnx regions_rect rect
   ```

3. 改了规则以后，读已有的框重新出计划、打分，不再推理：

   ```bash
   bash eval/tools/run_boxes.sh . seg1 rect                          # 分段，只开公式模型
   bash eval/tools/run_boxes.sh . seg2 rect $EVALSET/regions_rect    # 分段，两个模型都开
   bash eval/tools/mathrun_boxes.sh . math1 rect                     # 行内公式
   ```

4. 查看差别和错误：

   ```bash
   .venv/bin/python -I eval/tools/joindiff.py seg1 seg2      # 后一版新并上的单元
   .venv/bin/python -I eval/tools/mathscore.py math1 show    # 行内公式的错误，可以再加论文编号
   ```

### 回归语料

```bash
bash eval/regress/runplans.sh <改动前的代码副本> old
bash eval/regress/runplans.sh . new
.venv/bin/python -I eval/regress/segdiff.py old new     # 逐单元列出文本变了的地方
.venv/bin/python -I eval/regress/skipdiff.py old new    # 文本相同、跳过与否变了的单元
```

比较同一类模型的两个文件（如 fp32 和 int8）在回归语料上的差别：

```bash
bash eval/regress/compare_models.sh formula fp32=data/models/pix2text-mfd-1.5.onnx int8=data/models/pix2text-mfd-1.5.int8.onnx
bash eval/regress/compare_models.sh layout fp32=<fp32 版面模型> int8=<int8 版面模型>
```

输出单元数变了的文件，和同一位置上跳过与否变了的单元。

### int8 量化

公式检测模型：`run_cal.sh` 按指定的校准方法、页数和随机种子生成 int8 模型（`$WORK/int8/mfd.<名字>.onnx`），再出框、打分。

```bash
bash eval/models/run_cal.sh avg64:avg:64:0 mm64:minmax:64:0
```

方法可选 `minmax`、`avg`（各页最大最小值的滑动平均，产品里用的）、`percentile`、`entropy`。校准页默认从 `data/jobs` 下翻译过的 PDF 里抽，`CALIB_PDFS` 可以换目录，`EXCLUDE` 可以排除文件（逗号分隔的文件名，例如排除也在评测集里的论文）。`$WORK/int8/mfd.<名字>.onnx` 已存在时不再量化，可以用这种方式评测程序生成的模型文件。

版面模型：`lqcal.py` 用同样的设置量化，`run_layout_model.sh` 出框打分，`boxcmp.py` 逐类比较两套框（同类且 IoU ≥ 0.5 算对上）：

```bash
.venv/bin/python -I eval/models/lqcal.py . data/models/doclayout_yolo_docstructbench_imgsz1024.onnx $WORK/int8/layout.onnx
bash eval/models/run_layout_model.sh $WORK/int8/layout.onnx regions_lq8 rect
.venv/bin/python -I eval/models/boxcmp.py $EVALSET/regions_rect $EVALSET/regions_lq8
```

### 换公式检测模型

以 PDF-Extract-Kit 的公式检测模型为例。它只发布了 PyTorch 的 `.pt` 文件，要先在另一个虚拟环境里导出成 ONNX，导出用的依赖不装进项目的 `.venv`：

1. 下载 `yolo_v8_ft.pt`（地址见文末），核对 SHA-256。
2. `.pt` 是 pickle，加载时可以执行任意代码。加载前先用 `pklscan.py` 列出它引用的类，确认只有 torch、numpy 和 Ultralytics 的模型类：

   ```bash
   python3 -I eval/models/pklscan.py yolo_v8_ft.pt
   ```

3. 新建虚拟环境，装 CPU 版 torch、ultralytics、onnx、onnxslim，导出：

   ```bash
   YOLO_OFFLINE=1 <新环境>/bin/python -I eval/models/export.py yolo_v8_ft.pt
   ```

   `YOLO_OFFLINE=1` 让 Ultralytics 不联网检查更新。导出的 ONNX 是动态输入尺寸、不带 NMS 的，和 Pix2Text MFD 的格式相同，类别名也同为 `embedding`、`isolated`。

4. 按不同输入边长评测：

   ```bash
   bash eval/models/run_formula_model.sh yolo_v8_ft.onnx pek1024:1024 pek1280:1280
   ```

### 测速

在 WSL 里，一篇 24 页论文的分析耗时约 35 秒，几次之间的波动可达 ±5 秒（Windows 一侧的负载在 WSL 里看不到）。比较几种设置时要轮流跑、多跑几次取中位数，并且不能和其他评测同时跑。

| 脚本 | 测什么 |
|---|---|
| `models/qbench.py` | 几个公式检测模型逐页轮流推理，报每页耗时 |
| `models/bench2.py` | 几个模型或输入边长逐页轮流推理，报每页耗时 |
| `models/stages.py` | 翻译前的分析按程序报告的阶段（如分析版面、整理段落）分别计时，另计语言检测，并保存主线程的 cProfile |

## 2026-10-27 的结果

分析代码（`app/formats/`）和提交 749ad4c 相同，到 58cfce7 没有改动。公式检测模型的输入边长是 1024，页面图像只补到 32 的倍数。

| 设置 | 分段：拆错 / 并错 | 行内公式正确率 |
|---|---|---|
| 只用规则 | 92 / 0 | 78.4%（2522/3216） |
| 公式检测模型 fp32 | 89 / 0 | 98.9%（3152/3188） |
| 公式检测模型 int8，MinMax 校准，4 组抽页 | 89 / 0 | 98.5%～98.7% |
| 公式检测模型 int8，滑动平均校准，5 组抽页 | 89 / 0 | 98.9% |
| 同上，校准时排除评测集里的 2 篇，2 组抽页 | 89 / 0 | 98.9%（3150、3151/3186） |
| 公式检测模型 fp32 + 版面模型 fp32 | 73 / 0 | — |
| 公式检测模型 fp32 + 版面模型 int8 | 73 / 0 | — |
| PDF-Extract-Kit MFD，输入 1024 | 91 / 0 | 97.1%（3082/3173） |
| PDF-Extract-Kit MFD，输入 1280 | 89 / 0 | 98.8%（3133/3171） |

- 版面模型 int8 和 fp32 的框逐类比较，7946 个框中对上 98.6%。
- 6 核 i5-8500 上，公式检测模型每页推理 1.11 秒（fp32）、0.56 秒（int8）。
- 回归语料上，公式检测模型 int8（滑动平均）和 fp32 相比，约 2.1 万个单元中有 6 个跳过与否不同。

## 论文和文件来源

### 评测集（36 篇）

从 arXiv 15 个分类各取 3 篇：按提交时间倒序跳过最新的 300 篇，取 4～25 页、有 LaTeX 源码的论文。共下载 44 篇，其中 7 篇的源码里没有 `\documentclass`（2608.07601、2608.08170、2608.12396、2610.06091、2610.06149、2610.17315、2610.17341），1 篇（2610.17292）`gold.py` 选中的源文件里没有 `\begin{document}`，这 8 篇不用。

| arXiv 编号 | 分类 | 页数 | 标题 |
|---|---|---|---|
| [2608.05882v1](https://arxiv.org/abs/2608.05882v1) | q-bio.NC | 23 | Complexity and Stability of Neural Activity Across Aging and Neurodegenerative Disease |
| [2608.06149v1](https://arxiv.org/abs/2608.06149v1) | q-bio.NC | 24 | Errorless Irrationality: A unified computational account of the inverse base-rate effect across predictive, observational, and unsupervised procedures |
| [2608.06427v1](https://arxiv.org/abs/2608.06427v1) | econ.EM | 19 | Adversarial Causal Intervention Falsification |
| [2610.06279v1](https://arxiv.org/abs/2610.06279v1) | physics.optics | 8 | Electro-Optical Metalens for Near-Surface Cooling of a Levitated Particle |
| [2610.11173v1](https://arxiv.org/abs/2610.11173v1) | stat.ME | 16 | Bayesian inference under model misspecification |
| [2610.11349v1](https://arxiv.org/abs/2610.11349v1) | stat.ME | 23 | Sample-Efficient Generative Conformal Prediction |
| [2610.13095v1](https://arxiv.org/abs/2610.13095v1) | eess.SP | 10 | Stochastic Finite-Transition Learning and Exploration for Accelerated MRI Reconstruction |
| [2610.13108v1](https://arxiv.org/abs/2610.13108v1) | eess.SP | 5 | LLM-Assisted Hierarchical Interference Control for Satellite-Terrestrial Integrated Networks |
| [2610.13315v1](https://arxiv.org/abs/2610.13315v1) | eess.SP | 5 | JEPA-kMMVAE: CSI Prediction for Time-Varying Wireless Channels |
| [2610.18436v1](https://arxiv.org/abs/2610.18436v1) | math.PR | 10 | Stability of the Markov-Krein Transform Under Classical Convolutions |
| [2610.18440v1](https://arxiv.org/abs/2610.18440v1) | math.PR | 17 | A Central Limit Theorem for High-Temperature SYK Model |
| [2610.18445v1](https://arxiv.org/abs/2610.18445v1) | math.PR | 25 | The Singularity Has a Distribution: Exact Timing Laws and Detection Limits for a Stochastic Intelligence Explosion |
| [2610.18505v1](https://arxiv.org/abs/2610.18505v1) | astro-ph.GA | 10 | Black hole spin-mass correlation and accretion disks at birth stage of dense star clusters |
| [2610.18576v1](https://arxiv.org/abs/2610.18576v1) | astro-ph.GA | 19 | Massive double-exponential galaxies with extended LSB emission: An IllustrisTNG exploration II -are they unique? |
| [2610.18601v1](https://arxiv.org/abs/2610.18601v1) | astro-ph.GA | 13 | Dynamical formation of dormant black holes: Predictions for Gaia DR4 |
| [2610.20378v1](https://arxiv.org/abs/2610.20378v1) | hep-th | 11 | Origin of the red spectral tilt in Inflation without an Inflaton |
| [2610.20451v2](https://arxiv.org/abs/2610.20451v2) | hep-th | 11 | Analytical Time-Domain Waveforms for Black-Hole Scattering in the Self-Force Expansion |
| [2610.20452v1](https://arxiv.org/abs/2610.20452v1) | hep-th | 10 | Analytical Time-Domain Waveforms for Black-Hole Scattering in the Post-Minkowskian Expansion |
| [2610.23045v1](https://arxiv.org/abs/2610.23045v1) | math.AP | 21 | Isolated singularities of weak solutions to the Serrin-critical $p$-Laplace equation |
| [2610.23114v1](https://arxiv.org/abs/2610.23114v1) | math.AP | 12 | Linearization in nonlocal elasticity via Gamma-convergence |
| [2610.23115v1](https://arxiv.org/abs/2610.23115v1) | math.AP | 12 | Degenerated scalar balance laws: stability of the critical wave |
| [2610.25014v1](https://arxiv.org/abs/2610.25014v1) | quant-ph | 8 | Large Language Model Steering of Acousto-Optic Tweezer Arrays |
| [2610.25020v1](https://arxiv.org/abs/2610.25020v1) | quant-ph | 20 | Fast and High-Fidelity Motional-State-Preserving Transport in Optical Tweezers |
| [2610.25057v1](https://arxiv.org/abs/2610.25057v1) | quant-ph | 24 | Super Howe duality and rigidity of hybrid Fourier transforms |
| [2610.25476v1](https://arxiv.org/abs/2610.25476v1) | cs.RO | 8 | CusADi2: Latency-Optimized Batch Computation of Robot Kinematics, Jacobians, and Hessians on the GPU |
| [2610.25534v1](https://arxiv.org/abs/2610.25534v1) | cs.RO | 9 | AQUAACT: A Joystick-Level Underwater Teleoperation Dataset and Closed-Loop Benchmark |
| [2610.25585v1](https://arxiv.org/abs/2610.25585v1) | cs.RO | 7 | Adaptive Bendpoint Formation in Octopus-Inspired Soft Continuum Arms |
| [2610.27702v1](https://arxiv.org/abs/2610.27702v1) | cs.CL | 10 | Beyond Calibration: Temporal LLM Confidence Assessment Across Languages and Local Settings |
| [2610.27714v1](https://arxiv.org/abs/2610.27714v1) | cs.CL | 16 | When Does Experience Memory Help Closed-Loop LLM Driving Agents? A Controlled Evaluation with Resampled Warmup Pools, Token-Matched Fillers, and Audited Baselines |
| [2610.27720v1](https://arxiv.org/abs/2610.27720v1) | cs.CL | 11 | Tracing Recurrent and Emergent Hallucination Patterns in Decomposed Text-to-SQL Generation |
| [2610.28258v1](https://arxiv.org/abs/2610.28258v1) | cs.CV | 4 | Reliability-Aware Reconstruction of V-Shaped Surfaces with a Low-Cost Multi-Zone SPAD Sensor |
| [2610.28264v1](https://arxiv.org/abs/2610.28264v1) | cs.CV | 12 | PI-MLLM: Psychophysics-Inspired Multimodal Large Language Model for Oddly Satisfying Video Emotion Analysis |
| [2610.28280v1](https://arxiv.org/abs/2610.28280v1) | cs.CV | 19 | GALA: Geometry-Aware Language Model for Controllable Object Arrangement |
| [2610.29351v2](https://arxiv.org/abs/2610.29351v2) | cs.LG | 5 | Closed-Loop Dual-Reference Guidance for Accent Control in Cross-Lingual Voice Cloning |
| [2610.29355v1](https://arxiv.org/abs/2610.29355v1) | cs.LG | 16 | GaP-MoE : Explicit Expert Specialization in Mixture-of-Experts via Orthogonal Gradient Accumulation and Projection |
| [2610.29378v1](https://arxiv.org/abs/2610.29378v1) | cs.LG | 19 | Right On Q: Recovering Reasoning under Quantization via Self-Generated Trajectories |

### 回归语料（15 份）

| 文件名 | 页数 | 名称 | 来源 |
|---|---|---|---|
| 1312.6114 | 14 | Auto-Encoding Variational Bayes | <https://arxiv.org/abs/1312.6114> |
| 1406.2661 | 9 | Generative Adversarial Networks | <https://arxiv.org/abs/1406.2661> |
| 1412.6980 | 15 | Adam: A Method for Stochastic Optimization | <https://arxiv.org/abs/1412.6980> |
| 1502.03167 | 11 | Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift | <https://arxiv.org/abs/1502.03167> |
| 1706.03762 | 15 | Attention Is All You Need | <https://arxiv.org/abs/1706.03762> |
| 1810.04805 | 16 | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | <https://arxiv.org/abs/1810.04805> |
| 2006.11239 | 25 | Denoising Diffusion Probabilistic Models | <https://arxiv.org/abs/2006.11239> |
| 2310.06825 | 9 | Mistral 7B | <https://arxiv.org/abs/2310.06825> |
| 2501.12948 | 86 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | <https://arxiv.org/abs/2501.12948> |
| NIST.AI.100-1 | 48 | Artificial Intelligence Risk Management Framework (AI RMF 1.0) | <https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf> |
| ed3book_aug26 | 646 | Speech and Language Processing, 3rd edition draft (Jurafsky, Martin) | <https://web.stanford.edu/~jurafsky/slp3/ed3book_aug26.pdf> |
| lm_jan25 | 82 | N-gram Language Modeling (slides, Jurafsky) | <https://web.stanford.edu/~jurafsky/slp3/slides/lm_jan25.pdf> |
| implementing_data_cube | 25 | Implementing Data Cubes Efficiently (Harinarayan, Rajaraman, Ullman) | <https://doi.org/10.1145/233269.233333> |
| opt_views | 12 | Optimizing Queries Using Materialized Views: A Practical, Scalable Solution (Goldstein, Larson) | <https://dsg.uwaterloo.ca/seminars/notes/larson-paper.pdf> |
| phoenix-on-calcite-hadoop-summit-2016 | 32 | Cost-based Query Optimization (slides, Xue, Hyde, Hadoop Summit 2016) | <https://www.slideshare.net/HadoopSummit/costbased-query-optimization> |

- NIST.AI.100-1、ed3book_aug26、lm_jan25 三个链接下载到的文件和本地使用的文件 SHA-256 相同。
- opt_views 的链接是同一篇论文的另一份 PDF，页数相同，文件不同。
- implementing_data_cube 本地使用的是 25 页的长版本，来源未查到；链接是 SIGMOD 1996 的会议版（12 页）。
- 9 篇 arXiv 论文的链接指向最新版本，本地文件的版本以 SHA-256 为准。
- phoenix-on-calcite-hadoop-summit-2016 是 Hadoop Summit 2016 的幻灯片，SlideShare 下载需要登录。

### 模型

| 模型 | 用途 | 地址 |
|---|---|---|
| Pix2Text MFD 1.5（YOLO11m） | 公式检测，程序默认使用 | <https://huggingface.co/breezedeus/pix2text-mfd-1.5> |
| DocLayout-YOLO DocStructBench（YOLOv10m） | 版面分析 | <https://huggingface.co/wybxc/DocLayout-YOLO-DocStructBench-onnx> |
| PDF-Extract-Kit MFD（YOLOv8l） | 公式检测，比较用 | <https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0> 中的 `models/MFD/YOLO/yolo_v8_ft.pt`，SHA-256 `41029d5abb9b0b6df825ecaf8adf9151762ea0688ac0fc4ea0ea34ab9a5808fc` |
