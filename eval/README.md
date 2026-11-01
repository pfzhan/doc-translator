# Evaluating PDF translation

English | [简体中文](README.zh-CN.md)

The scripts here measure the analysis done before a PDF is translated:

- Segmentation: whether paragraph boundaries are cut correctly.
- Inline formulas: whether formulas in body text become whole placeholders instead of being sent for translation.
- The boxes of the two models (formula detection, layout analysis): how much they change when the model is
  replaced or quantized.

The gold standards for segmentation and inline formulas come from the LaTeX sources of arXiv papers (the
evaluation set, 36 papers). In addition, 15 PDFs without sources form the regression corpus, used to list the
differences in output between two versions for a person to judge.

## Contents

| Path | Contents |
|---|---|
| `env.sh` | Path settings shared by the scripts |
| `papers.tsv` | Papers of the evaluation set: arXiv id (with version), category, page count, SHA-256 of the PDF, title |
| `tools/` | Evaluation set: downloading papers, generating gold standards, producing plans, scoring |
| `regress/` | Regression corpus: producing plans and comparing two versions; `files.tsv` lists the files |
| `models/` | Model experiments: int8 quantization, replacing the formula detection model, timing |

Below, a plan is the list of translation units the program cuts from a PDF; each unit records the page number,
whether it is skipped (keeps the original) and the text sent for translation. Producing a plan runs only the
analysis before translation and never calls a translation service. Each run of plans gets a tag, plans are stored
as `<id>.<tag>.json`, and the plans of two tags can be compared.

## Setup

### Environment

- The project's `.venv` with `requirements.txt` and onnxruntime installed. Generating int8 models also needs
  onnx: `.venv/bin/pip install onnx`.
- The two model files are downloaded on the program's 版面模型 (Layout models) page and saved to `data/models/`
  by default:
  - formula detection model `pix2text-mfd-1.5.onnx`
  - layout model `doclayout_yolo_docstructbench_imgsz1024.onnx`
- The scripts run with bash, for example `bash eval/tools/run.sh . rules`.

The shell scripts read their paths from `env.sh`; these environment variables override the defaults:

| Variable | Meaning | Default |
|---|---|---|
| `EVALSET` | Evaluation data directory for papers, gold standards, plans and boxes | `~/doc-translator-evalset` |
| `PY` | Python with the project dependencies | `.venv/bin/python` in the project directory |
| `CODE` | Code directory used to produce plans and boxes; may point to a copy of a commit | the project directory |
| `REGRESS_PDFS` | Directory of the regression corpus PDFs | `data/jobs` in the project directory |
| `WORK` | Directory for experiment outputs (int8 models and so on) | `$EVALSET/work` |

Layout of the evaluation data directory:

| Subdirectory | Contents |
|---|---|
| `raw/<id>/` | Papers: `paper.pdf`, LaTeX source `src/`, `meta.txt` |
| `gold/`, `mathgold/` | Gold standards for segmentation and inline formulas |
| `mfd/` | Formula detection boxes, `<id>.<tag>.json` |
| `regions_<name>/` | Layout model boxes, one `<id>.json` per paper |
| `plans/`, `mplans/` | Segmentation plans, plans with formula text |
| `results/` | Per-paper segmentation scores |
| `regress/` | Boxes (`mfd/`, `regions_<name>/`) and plans (`plans/`) of the regression corpus |

### Downloading the papers

```bash
mkdir -p $EVALSET/raw
.venv/bin/python -I eval/tools/fetch.py eval/papers.tsv $EVALSET/raw
```

The script downloads the PDF and LaTeX source of the versions in the list, extracting only `.tex`, `.bbl`,
`.sty` and `.cls` from the source. arXiv asks for at least 3 seconds between requests, so the 36 papers take
about 4 minutes. The downloaded files come from outside: the evaluation scripts only read the sources as text
and never compile them, and Python always runs with `-I`, so no modules are loaded from the current directory or
the script's directory. The script reports a downloaded PDF whose SHA-256 differs from the list.

### Generating the gold standards

```bash
mkdir -p $EVALSET/gold $EVALSET/mathgold
for d in $EVALSET/raw/*/; do id=$(basename $d)
  .venv/bin/python -I eval/tools/gold.py $d $EVALSET/gold/$id.json
  .venv/bin/python -I eval/tools/mathgold.py $d $EVALSET/mathgold/$id.json
done
```

The output of both scripts depends only on the source, so the same source always gives the same result.

### Regression corpus

`regress/files.tsv` lists the name, page count, SHA-256 and source of the 15 files (see the end of this
document). Put these PDFs in one directory and point `REGRESS_PDFS` to it. Versions downloaded online may differ
from those in the list; the SHA-256 is authoritative.

## Gold standards and scoring

### Segmentation

`gold.py` takes paragraphs from the source: paragraphs break at blank lines, `\par`, sectioning commands, block
environments and `\item`; the two sides of a display formula inside a paragraph are ignored boundaries, so
splitting there or not are both accepted. The 36 papers have 2073 paragraphs and 1689 paragraph boundaries.

`score.py` locates the start and end of each paragraph in the plan (by 4-word fragments within 12 words of the
paragraph start and end that occur only once in the whole text) and counts two kinds of errors:

- Wrong split: the start and end of one paragraph fall in different units.
- Wrong join: the end of one paragraph and the start of the next fall in the same unit.

The gold standard does not cover bibliographies; their segmentation is listed with `joindiff.py` and checked by
hand.

### Inline formulas

`mathgold.py` records every `$…$` and `\(…\)` formula in the source with 4 body words on each side (anchors).
`mathscore.py` uses the anchors to locate each formula in the plan and records it as:

| Label | Meaning |
|---|---|
| `full` | The whole formula is in a placeholder |
| `minor` | Only whole words such as numbers or units (like 50 Myr) are left outside the placeholder, which does not hurt translation |
| `bad_partial` | A variable, Greek letter or math symbol is left outside the placeholder |
| `bad_missed` | No placeholder was made |
| `empty` | Nothing comparable between the two anchors |
| `lost` | Anchors not found; not scored |

Accuracy = (`full` + `minor` + `empty`) / the number scored. Anchor words turned into placeholders are counted as
false positives (`fp` in the output). Results are split by body font into two groups: papers set in Computer
Modern (`cm`) and in other fonts (`other`, such as Times). In the second group, italic variables use the same
font as italic body text and are harder to tell apart.

## Running the evaluation

All commands below run in the project directory with the data directory set:

```bash
export EVALSET=~/doc-translator-evalset WORK=~/doc-translator-evalset/work
```

The first argument `.` is the code directory used to produce plans; pointing it to a copy of another commit
compares two versions.

### Rules only

```bash
bash eval/tools/run.sh . rules
```

This produces and scores segmentation plans, writing per-paper results to `results/rules.txt` with the total on
the last line. For inline formulas, use `mathrun.sh`:

```bash
bash eval/tools/mathrun.sh . rules
```

### With models

Model inference is slow, so the boxes are saved first, and after rule changes plans are produced again from the
saved boxes only.

1. Produce boxes for the 36 papers with the formula detection model and score them; the boxes are saved as
   `mfd/<id>.rect.json`:

   ```bash
   bash eval/models/run_formula_model.sh data/models/pix2text-mfd-1.5.onnx rect:1024
   ```

2. Produce boxes with the layout model into `$EVALSET/regions_rect/`, then score segmentation together with the
   formula boxes from step 1:

   ```bash
   bash eval/models/run_layout_model.sh data/models/doclayout_yolo_docstructbench_imgsz1024.onnx regions_rect rect
   ```

3. After changing rules, produce and score plans again from the existing boxes without inference:

   ```bash
   bash eval/tools/run_boxes.sh . seg1 rect                          # segmentation, formula model only
   bash eval/tools/run_boxes.sh . seg2 rect $EVALSET/regions_rect    # segmentation, both models
   bash eval/tools/mathrun_boxes.sh . math1 rect                     # inline formulas
   ```

4. Look at differences and errors:

   ```bash
   .venv/bin/python -I eval/tools/joindiff.py seg1 seg2      # units newly joined in the second version
   .venv/bin/python -I eval/tools/mathscore.py math1 show    # inline formula errors; a paper id can be added
   ```

### Regression corpus

```bash
bash eval/regress/runplans.sh <copy of the code before the change> old
bash eval/regress/runplans.sh . new
.venv/bin/python -I eval/regress/segdiff.py old new     # where the text changed, unit by unit
.venv/bin/python -I eval/regress/skipdiff.py old new    # units with the same text but a different skip flag
```

To compare two files of the same kind of model (such as fp32 and int8) on the regression corpus:

```bash
bash eval/regress/compare_models.sh formula fp32=data/models/pix2text-mfd-1.5.onnx int8=data/models/pix2text-mfd-1.5.int8.onnx
bash eval/regress/compare_models.sh layout fp32=<fp32 layout model> int8=<int8 layout model>
```

This lists files whose unit count changed and units at the same position whose skip flag changed.

### int8 quantization

Formula detection model: `run_cal.sh` generates an int8 model (`$WORK/int8/mfd.<name>.onnx`) with the given
calibration method, page count and random seed, then produces boxes and scores them.

```bash
bash eval/models/run_cal.sh avg64:avg:64:0 mm64:minmax:64:0
```

Methods are `minmax`, `avg` (moving average of per-page minimums and maximums, the one the application uses),
`percentile` and `entropy`. Calibration pages are sampled from the translated PDFs under `data/jobs` by default;
`CALIB_PDFS` sets another directory, and `EXCLUDE` leaves files out (comma-separated file names, for example
papers that are also in the evaluation set). When `$WORK/int8/mfd.<name>.onnx` already exists it is not quantized
again, which allows evaluating a model file generated by the application.

Layout model: the application's Layout models page generates the int8 version directly (`.int8.onnx` under
`data/models/`); for evaluating calibration methods use `lqcal.py`, which has the same settings as the
application and also accepts `EXCLUDE`. `run_layout_model.sh` produces and scores boxes, and `boxcmp.py` compares
two sets of boxes class by class (same class and IoU ≥ 0.5 counts as a match):

```bash
.venv/bin/python -I eval/models/lqcal.py . data/models/doclayout_yolo_docstructbench_imgsz1024.onnx $WORK/int8/layout.onnx
bash eval/models/run_layout_model.sh $WORK/int8/layout.onnx regions_lq8 rect
.venv/bin/python -I eval/models/boxcmp.py $EVALSET/regions_rect $EVALSET/regions_lq8
```

### Replacing the formula detection model

Taking the formula detection model of PDF-Extract-Kit as an example: it is published only as a PyTorch `.pt`
file, which must first be exported to ONNX in a separate virtual environment, keeping the export dependencies
out of the project's `.venv`:

1. Download `yolo_v8_ft.pt` (see the end of this document for the URL) and check its SHA-256.
2. A `.pt` file is a pickle and can run arbitrary code when loaded. Before loading it, list the classes it
   references with `pklscan.py` and confirm they are only torch, numpy and Ultralytics model classes:

   ```bash
   python3 -I eval/models/pklscan.py yolo_v8_ft.pt
   ```

3. Create a new virtual environment with the CPU version of torch, ultralytics, onnx and onnxslim, and export:

   ```bash
   YOLO_OFFLINE=1 <new environment>/bin/python -I eval/models/export.py yolo_v8_ft.pt
   ```

   `YOLO_OFFLINE=1` keeps Ultralytics from checking for updates online. The exported ONNX has a dynamic input
   size and no NMS, the same format as Pix2Text MFD, with the same class names `embedding` and `isolated`.

4. Evaluate at different input sides:

   ```bash
   bash eval/models/run_formula_model.sh yolo_v8_ft.onnx pek1024:1024 pek1280:1280
   ```

### Timing

Under WSL, analyzing a 24-page paper takes about 35 seconds, varying by up to ±5 seconds between runs (load on
the Windows side is not visible inside WSL). When comparing settings, run them in turn, several times, take the
median, and never run other evaluations at the same time.

| Script | Measures |
|---|---|
| `models/qbench.py` | Several formula detection models inferring in turn page by page; time per page |
| `models/bench2.py` | Several models or input sides inferring in turn page by page; time per page |
| `models/stages.py` | Time of each stage of the analysis before translation, as reported by the program (such as analyzing layout and organizing paragraphs), plus language detection; saves a cProfile of the main thread |

## Results of 2026-10-27

The analysis code (`app/formats/`) is the same as in commit 749ad4c and unchanged up to 58cfce7. The formula
detection model uses an input side of 1024, with page images padded only to a multiple of 32.

| Setting | Segmentation: wrong splits / wrong joins | Inline formula accuracy |
|---|---|---|
| Rules only | 92 / 0 | 78.4% (2522/3216) |
| Formula detection fp32 | 89 / 0 | 98.9% (3152/3188) |
| Formula detection int8, MinMax calibration, 4 page samples | 89 / 0 | 98.5% to 98.7% |
| Formula detection int8, moving average calibration, 5 page samples | 89 / 0 | 98.9% |
| Same, excluding 2 papers of the evaluation set from calibration, 2 page samples | 89 / 0 | 98.9% (3150, 3151/3186) |
| Formula detection fp32 + layout fp32 | 73 / 0 | — |
| Formula detection fp32 + layout int8 (generated by `lqcal.py`, excluding 2 papers of the evaluation set) | 73 / 0 | — |
| Formula detection fp32 + layout int8 (generated by the application) | 73 / 0 | — |
| PDF-Extract-Kit MFD, input 1024 | 91 / 0 | 97.1% (3082/3173) |
| PDF-Extract-Kit MFD, input 1280 | 89 / 0 | 98.8% (3133/3171) |

- Comparing the int8 and fp32 boxes of the layout model class by class, 98.6% of 7946 boxes match (for both int8
  files); captions match 96.8% to 97.3%, the lowest of all classes.
- On a 6-core i5-8500, layout model inference takes 1.05 seconds per page (fp32) and 0.66 seconds (int8).
- On a 6-core i5-8500, formula detection inference takes 1.11 seconds per page (fp32) and 0.56 seconds (int8).
- On the regression corpus, compared with fp32, int8 changes the skip flag of 6 of about 21,000 units for the
  formula detection model (moving average) and 6 for the layout model.

## Papers and file sources

### Evaluation set (36 papers)

Three papers from each of 15 arXiv categories: skipping the newest 300 submissions by submission time, papers of
4 to 25 pages with LaTeX source. 44 papers were downloaded; 7 of them have no `\documentclass` in their source
(2608.07601, 2608.08170, 2608.12396, 2610.06091, 2610.06149, 2610.17315, 2610.17341), and for 1 (2610.17292) the
source file chosen by `gold.py` has no `\begin{document}`. These 8 are not used.

| arXiv id | Category | Pages | Title |
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

### Regression corpus (15 files)

| File name | Pages | Title | Source |
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

- The files downloaded from the NIST.AI.100-1, ed3book_aug26 and lm_jan25 links have the same SHA-256 as the
  local files.
- The opt_views link is another PDF of the same paper, with the same page count but a different file.
- implementing_data_cube is used locally in a 25-page long version of unknown origin; the link is the SIGMOD 1996
  conference version (12 pages).
- The links of the 9 arXiv papers point to the latest version; the version of the local files is identified by
  SHA-256.
- phoenix-on-calcite-hadoop-summit-2016 is a slide deck from Hadoop Summit 2016; downloading from SlideShare
  requires logging in.

### Models

| Model | Use | URL |
|---|---|---|
| Pix2Text MFD 1.5 (YOLO11m) | Formula detection, used by the application by default | <https://huggingface.co/breezedeus/pix2text-mfd-1.5> |
| DocLayout-YOLO DocStructBench (YOLOv10m) | Layout analysis | <https://huggingface.co/wybxc/DocLayout-YOLO-DocStructBench-onnx> |
| PDF-Extract-Kit MFD (YOLOv8l) | Formula detection, for comparison | `models/MFD/YOLO/yolo_v8_ft.pt` in <https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0>, SHA-256 `41029d5abb9b0b6df825ecaf8adf9151762ea0688ac0fc4ea0ea34ab9a5808fc` |
