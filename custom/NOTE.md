# Replicating Qwen3-Omni-30B-A3B-Instruct on EvaOmni — implementation notes

Goal: reproduce Table 2 (Acc. / C.G. / B.G.) of the paper for
Qwen3-Omni-30B-A3B-Instruct, locally, without modifying any original repo file
or `data/`. New code is in `custom/`; outputs go to `custom/output/`.

Paper reference values:

| | Acc. | C.G. | B.G. |
|---|---:|---:|---:|
| Level 1 | 28.6 | 5.1 | 0.6 |
| Level 2 | 22.4 | 3.4 | 0.4 |
| Level 3 | 22.2 | 2.4 | 0.1 |
| Overall | 25.2 | 4.0 | 0.4 |

## Step 0 — scoring (`custom/score.py`)

- The metrics come from the authors' code. `compute_grounded_scores` and
  `parse_evidence_interval` from `scripts/eval_example.py`, and
  `parse_model_output` from `scripts/eval_gemini.py`, are **imported** by file
  path with `importlib`, not copied. Importing `eval_gemini.py` needs `requests`.
- Input format (written by our adapters): a JSONL file, one line per attempt,
  with `qa_id`, `status` (`"ok"` means inference returned text) and
  `raw_response`. The last line for each qa_id wins.
- **Strict** mode uses `eval_example.parse_response`. It is all-or-nothing: valid
  JSON, `answer` exactly one of A–H, both evidence lists present, every
  timestamp `[mm:ss,mm:ss]` with end > start. If anything is malformed, the
  question scores zero on all metrics, including Acc.
- **Permissive** mode uses `eval_gemini.parse_model_output`, which accepts
  fenced JSON, JSON embedded in prose, or a bare letter as a last resort. Each
  interval is then read with `parse_evidence_interval` (accepts `m:ss`,
  `hh:mm:ss`, decimals). Unreadable intervals are dropped one by one, and the
  remaining intervals and the letter still count.
- Missing questions and failed inference score zero and stay in the
  denominator (n = 1027), as in paper App. D.3.
- Accuracy CI: Wilson 95% interval, binomial over the questions of each
  group. No CI is computed for C.G. and B.G. **Caveat:** this interval measures
  variability due to *sampling the questions*. We evaluate on the *same*
  questions as the paper, so it is NOT the right yardstick for judging the
  replication. The right one is our own **run-to-run variability** (see
  "Run-to-run variability" below). The Wilson interval is useful only for
  comparing against baselines, or for generalising beyond these 1027 questions.
- Permissive answer sources (diagnostic). `eval_gemini.parse_model_output`
  takes the first isolated A–H, ignoring case. In free text, the English
  article "a" therefore becomes answer "A". The report splits permissive
  answers into `json_letter` (the JSON "answer" field is exactly one letter),
  `json_nonstandard` (JSON whose "answer" field is longer text; the letter is
  taken from that field), `free_text` (no JSON at all; the letter comes from
  the raw text) and `none`. For each group it gives n, the letter
  distribution and the accuracy within the group.
  `permissive_no_free_text` is the permissive score with `free_text`
  recoveries set to zero (denominator still 1027). The original function is
  not modified.
- Letter distribution: always saved and printed, overall and per level, for
  ground truth, strict predictions and permissive predictions
  (`none` = no usable letter).
- Baselines: always "A", and chance = mean over questions of 1/n_options.
  Options are counted with the regex `^[A-Z]\. `.
- Self-test (`custom/selftest_score.py`): annotations as predictions give 100
  everywhere; a wrong letter with perfect evidence gives 0 everywhere; missing
  rows keep the denominator at 1027; parser edge cases behave as expected.

### Dataset facts that matter for interpretation

- 1025 of 1027 questions have **8 options (A–H)**. qa 209 has 9 (A–I) and qa 483
  has 6. The overall chance level is therefore **12.5%**, not 25% as said in
  §3.1 of the paper. qa 649 looks like 16 lines but has 8 options with line
  breaks inside them.
- Both authors' parsers accept only A–H, so for qa 209 option "I" can never be
  scored, but its correct answer is "A" and is unaffected.
- Answer letters are unbalanced: "A" is correct in 19.2% of the questions. By
  level: L1 11.7%, **L2 27.4%**, L3 17.0%. "Always A" scores higher than the
  paper's Qwen3-Omni-Instruct on Level 2 (22.4%) and higher than Qwen2.5-Omni-7B
  overall (16.4%).
- All reference timestamps are in strict `[mm:ss,mm:ss]` format, and every
  question has at least one audio and one visual evidence interval.
- `data/resized_video/` is already 1 FPS, with the longer side at 480 px (e.g.
  480x270) and AAC 48 kHz audio.

### Possible differences from the paper (scoring)

- The paper does not say whether its numbers are strict or permissive; we
  report both. App. D.3 says that "comma-based decimal timestamps are
  normalized numerically". It is not clear what this means, and neither of our
  modes does it explicitly.
- Permissive mode inherits the authors' letter fallback (`\b[A-H]\b` on the
  raw text). On prose output it can pick up a spurious "A" (e.g. "A man…").
  See `answer_sources` and `permissive_no_free_text` in the report.

## Run-to-run variability (plan)

The model card has two decoding settings:
- Section "Setting for Evaluation": **Instruct models use greedy decoding**,
  no system prompt, text after the multimodal input, video at fps=2 (the paper
  uses 1 FPS).
- The vLLM code snippets: sampling with temperature=0.6, top_p=0.95, top_k=20,
  max_tokens=16384, `LLM(seed=1234)`.

The paper only says "settings recommended in their model cards", so we do not
know which one the authors used. Approved plan, in this order:
1. Full run (1027) with greedy decoding, the most likely setting since the
   model card prescribes greedy for evaluating Instruct models. It is compared
   with the paper first.
2. Full run (1027) with the snippet's sampling: temperature=0.6, top_p=0.95,
   top_k=20, seed=1234.
3. Repeats on `custom/subsets/sub200.json`: 200 questions, stratified 85/85/30,
   drawn once with `random.Random(20261113)` and frozen for future
   experiments.
   - Greedy repeated identically: measures vLLM non-determinism.
   - Sampling repeated with a DIFFERENT seed (4321): measures sampling
     variability. The same seed would only measure non-determinism.
   - Report the per-metric differences between repeats and the question-level
     agreement (fraction of identical letters).

Seeds: `LLM(seed=seed)` as in the model card. In addition, each sampled request
gets `SamplingParams(seed=sha256(f"{seed}-{qa_id}"))`, so that a question's
sample does not depend on batching or on which GPU it lands on.

A difference from the paper is meaningful only if it is clearly larger than
this run-to-run spread.

## Step 1 — engine and adapter (`custom/run_qwen_omni.py`)

- Engine: vLLM 0.17.1 (in `av_env`, see below), offline `LLM` class, V1 engine. The model
  card's `VLLM_USE_V1=0` and its vLLM fork (0.9.2-based) are obsolete;
  upstream support is in `vllm/model_executor/models/qwen3_omni_moe_thinker.py`.
- Text-only output: vLLM maps `Qwen3OmniMoeForConditionalGeneration` to
  `Qwen3OmniMoeThinkerForConditionalGeneration` and skips the `talker.` and
  `code2wav.` weights. No speech component is loaded.
- Audio of the video: as in the model card, `qwen_omni_utils.process_mm_info(
  messages, use_audio_in_video=True)` extracts the audio track (PyAV,
  resampled to 16 kHz with librosa). The audio is passed as a separate `audio`
  item, with `mm_processor_kwargs={"use_audio_in_video": True}`. vLLM checks
  that the audio and video items are paired, and interleaves them in time.
- FPS and time encoding: in vLLM 0.17.1 the M-RoPE time step for video is
  `second_per_grid_ts * position_id_per_seconds(13)`. vLLM falls back to
  `second_per_grid_ts = 2.0` when the key is absent, and the HF processor
  outputs a key with a different name (`video_second_per_grid`), so the
  fallback is what is used. With `temporal_patch_size = 2`, 2.0 s/grid
  corresponds exactly to **1 FPS**, which matches `resized_video`. To be
  verified empirically in step 2 (grid_t × 2 = number of frames).
- `qwen_omni_utils` defaults to FPS=2.0. On our 1 FPS files it clamps to all
  frames. We pass `fps: 1.0` explicitly in the message. The number of frames is
  floored to an even number (FRAME_FACTOR=2): with an odd frame count the last
  frame (last second) is dropped and `sample_fps` becomes e.g. 0.9966 (video
  179: 293 frames -> 292). vLLM still assumes exactly 2.0 s per temporal grid,
  an error of at most 1 s at the very end of the video.
- The HF video processor in transformers 4.57.6 has `do_sample_frames=False`,
  so the frames from qwen_omni_utils are not resampled again.
- Frame size: qwen_omni_utils is called as in the model card, i.e. with the
  default `image_patch_size=14` (factor 28): 480x270 -> 476x280, bicubic. The
  HF processor inside vLLM then resizes again to multiples of 32 (patch 16,
  merge 2). This double resize is what the model card's pipeline does. Measured
  token counts are in the `input` part of each record of `output.jsonl`.
- No system prompt (the chat template adds none by default). The message order
  is video first, then the text. The prompt is `PROMPTS["video_audio"]` from
  `prompts.py`, filled with `.format(question=..., options=...)` and no strip
  (no record has leading or trailing whitespace).
- `limit_mm_per_prompt={"image": 0, "video": 1, "audio": 1}` (the model card has
  3/3/3). This only affects memory pre-allocation, not outputs.
- Failures: a batch that raises is retried one request at a time. A failing
  request is recorded with `status=error`; prompts longer than
  `max_model_len` get `error_type=prompt_too_long`. They score zero and stay in
  the denominator. `--resume` retries errors, except `prompt_too_long`.
- Context: `max_model_len = 65536` (= `max_position_embeddings`; the model
  card uses it for 4-GPU serving, and 32768 for 1 GPU). Rough estimate: about
  65 video tokens/s at 480x270 plus the audio tokens, so a 12-minute video
  needs about 55k+ tokens and would not fit in 32768. To be measured in step 2.
- Parallelism: 4 independent replicas (one per GPU, TP=1, data parallel over
  qa_id shards). The bf16 thinker is about 62 GB, which leaves about 30 GB of
  KV cache per GPU.
- Model revision pinned: `26291f793822fb6be9555850f06dfe95f2d7e695`.

## Environment `av_env` (`custom/env/`)

- New conda env, independent of `vllm_env`, built from `vllm_env`'s pip freeze
  (`custom/env/av_env_base_requirements.txt`, the same as the freeze except the
  non-portable `packaging @ file://` line, replaced by `packaging==25.0`). The
  recipe is in `custom/env/create_av_env.sh`.
- Same versions as vllm_env: Python 3.12.13, vllm 0.17.1, torch 2.10.0+cu128,
  transformers 4.57.6, numpy 2.2.6 (and every other pinned package). Python
  comes from conda-forge rather than defaults: same version, different build.
- Added: qwen-omni-utils 0.0.9, librosa **0.11.0**, audioread, soundfile, and
  ffmpeg 9.0.2 (conda-forge).
  - librosa must be < 1.0: qwen_omni_utils passes `audioread.FFmpegAudioFile`
    objects to `librosa.load`, and librosa 1.0 removed audioread support.
  - audioread is an undeclared dependency of qwen_omni_utils.
  - audioread decodes the audio with the `ffmpeg` binary. `vllm_env` has no
    ffmpeg, so the model-card pipeline could not read the video audio there.
    `run_qwen_omni.py` puts the interpreter's `bin/` first in PATH.
- `pip check` reports only "decord 0.6.0 is not supported on this platform".
  The same message appears in vllm_env: the decord wheel is tagged cp36, but it
  imports and reads our videos correctly. decord is the video reader that
  qwen_omni_utils selects.

## Organisation

- `custom/output/<run_name>/` holds one folder per run (since 2026-11-16):
  - `info.txt`: run information and settings (see below);
  - `output.jsonl`: one record per answered question, written during the run
    (append + fsync, the last record per qa_id wins, as in the repo's
    `eval_example.py`). Each record is pretty-printed JSON over several lines:
    qa_id, video_id, level, capability, time, status (+ error_type/error),
    `input` (video_seconds, frames, frame size, sample_fps, audio_seconds,
    audio_peak, preprocess_s, prompt/video/audio/text tokens,
    video_tokens_per_temporal_grid, context_fraction) and `output`
    (stop_cause, n_output_tokens, raw_response). Values shared by all
    questions (sampling params, max_model_len) are only in info.txt. Several
    workers write the same file under a lock;
  - `evaluation/` (written by score.py): `results.json` (summary per mode and
    level with all metrics and 95% CI, by_capability, baselines, diagnostics,
    input statistics, one entry per question with ground truth, prediction and
    scores in every mode) and `evaluation.txt` (see below).
  There is no logs/ folder: worker output goes to the terminal; the shard of
  each worker is passed on its command line.
  `custom/output/RUNS.md` is the index. If the name already exists, the run
  goes to `_v2`, `_v3`, ...
- Runs made before 2026-11-16 have responses_w*.jsonl, input_stats_w*.jsonl,
  scores.json, paper_comparison.md, logs/, config.json, env.txt. The code still
  reads them (`runinfo.load_outputs`, `runinfo.read_run_meta`,
  `score.run_scores`), so they can be re-scored, resumed and compared.
- info.txt starts with a `[run]` and a `[settings]` section (one
  `key : <JSON>` line each, `custom/runinfo.py`); `--resume <exact folder>`
  requires identical `[settings]` (idea from the repo's
  `eval_example.py` run_config.json). Resume / finish / scoring lines are
  appended below. Runs made before 2026-11-16 have config.json + env.txt
  instead; they are still read (fallback in `runinfo.read_run_meta`).
- Since 2026-11-16 no tool writes `_`-folders in `custom/output/`: the scorer
  self-test uses a temporary folder, `test_maxlen.py` and `compare_runs.py`
  only print. Folders like `_failed/`, `_superseded/`, `_comparisons/`,
  `_logs/`, `_selftest/`, `_tests/` are left over from earlier work.
- `custom/media/muted/` contains copies of the videos with the audio track
  kept but silent (`custom/tools/make_muted_videos.py`). The video stream is
  copied bit-for-bit, and the silent audio has the same sample rate, channel
  layout and duration as the original audio track. Some originals have audio
  up to 0.3 s longer than the video, so `-shortest` would have shortened the
  audio. Every file is verified (identical decoded frames, all samples zero,
  audio duration within 0.05 s) and listed in `manifest.json`.
- `custom/subsets/` holds frozen qa_id lists (`custom/tools/make_subsets.py`,
  which never overwrites): `sub200`, `check10-l1`, `check50`, `onepervideo`.

## Problems found while bringing the adapter up (step 2)

1. **fork vs spawn**: CUDA is initialised in the worker before vLLM starts its
   engine-core subprocess. The worker sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
   The crashed run is archived in `custom/output/_failed/..._crash-fork`.
2. **vLLM 0.17.1 bug with `use_audio_in_video` and the multimodal processor
   cache** (`vllm/multimodal/cache.py::get_and_update_item` +
   `qwen3_omni_moe_thinker.py::_maybe_apply_prompt_updates`). When the same
   video is seen again, the cache returns the video item as `None`. The check
   `if item and item["use_audio_in_video"].data` then sets
   `use_audio_in_video=False`, the audio placeholder cannot be applied, and an
   `AssertionError: Failed to apply prompt replacement for mm_items['audio'][0]`
   is raised. **This affects every question after the first one on the same
   video** (2.6 questions per video on average). It fails loudly rather than
   silently. Workaround: `mm_processor_cache_gb=0`, so every request
   re-processes its video. KV prefix caching stays enabled. In addition, every
   response is checked: a prompt with 0 audio or 0 video tokens is recorded as
   `error_type=mm_tokens_missing`. The model card's own vLLM example would hit
   this bug on any dataset with several questions per video, unless their
   version did not cache.
3. **CUDA OOM in the vision encoder** with `gpu_memory_utilization=0.95` (the
   model card's value) on a 577 s video. Model weights take 59.4 GiB; at 0.95
   the KV cache gets 28.9 GiB and there is not enough room left for the ViT
   activations of about 290 frame-pairs. We use **0.80** plus
   `PYTORCH_ALLOC_CONF=expandable_segments:True`. This is a memory setting only,
   so outputs are unaffected. If the engine dies, the worker exits WITHOUT
   recording the remaining questions, so `--resume` retries them instead of
   scoring them as failures. Archived in `_failed/..._crash-oom-cachebug`.
4. **vLLM 0.17.1 bug in `merge_interleaved_embeddings`**
   (`qwen2_5_omni_thinker.py`, called by Qwen3-Omni when
   use_audio_in_video=True). The function decides which embedding is video and
   which is audio only from token counts, as if the step held a single
   request. With `max_num_seqs=8`, chunked prefill put parts of two requests
   in one step: the audio of request A, which is smaller than A's video, was
   classified as video, and the engine crashed with
   `shape mismatch: value tensor of shape [10064, 2048] cannot be broadcast to
   indexing result of shape [13520, 2048]` (stats run, after 56/392 videos).
   In rare size coincidences the mix-up could even be silent. Fix:
   **`max_num_seqs=1`**, one request per engine step. Within a single request
   the order is always [video slice, audio slice], which is handled
   correctly. This is also a run setting (`[settings]`), so `--resume` cannot mix
   values. With one request per step, batch composition is no longer a source of variation; any residual non-determinism is measured by the greedy repeat on sub200.
   Note: comparing the superseded check10 (8 requests together) with the new one (one at a time) does NOT measure non-determinism, because the batched run could have been altered by this very bug. The real measure is the sub200 greedy repeat.
   Cost: little, since the time is dominated by prefill (prompts of 10k–62k
   tokens, outputs of about 70 tokens). All runs made before this fix
   (check10 x2, a partial check50, partial stats) are in
   `custom/output/_superseded/` and must not be used.

## Long videos: the 768-frame cap (found in step 2b)

`qwen_omni_utils` caps videos at `FPS_MAX_FRAMES = 768`. A video longer than
768 s (e.g. video 036, 779 s) is uniformly subsampled to 768 frames
(`sample_fps` 0.986). vLLM, however, assumes exactly 2.0 s per temporal grid
(1 FPS) for the time encoding, so the model's internal clock for such videos
drifts by up to about 11 s at the end. The audio is not subsampled and keeps
its correct timing, so audio and video positions disagree slightly near the
end. This is what the model-card pipeline does with these files, so we keep
it, but it is a possible source of grounding error on the longest videos. The
number of affected videos and questions is in the step-2b stats run.

## Step 2 — results of the three checks (runs with max_num_seqs=1)

### (a) Audio reaches the model — PASSED
`qwen3omni-instr_av_greedy_check10` vs `qwen3omni-instr_av-mute_greedy_check10`
(10 Level-1 questions; comparison in
`custom/output/_comparisons/..._check10__vs__..._check10.json`):
- Raw responses identical: 0/10; same letter: 8/10.
- Identical prompt structure: same number of video tokens 10/10 and of audio
  tokens 9/10 (the remaining one differs by 1 token because of AAC padding).
  The only thing that changes is the audio content (peak about 0.5–1.0 in the
  normal run, 0 in the muted run).
- With silent audio the model often cites `[00:00,00:01]` as evidence.
  Acc. is 2/10 in both runs (meaningless on 10 questions); C.G. 10 -> 0,
  B.G. 2.5 -> 0.
- Audio rate: exactly 13.0 audio tokens per second on every video.

### (b) Input sizes and truncation (`qwen3omni-instr_av_greedy_onepervideo_stats`, all 392 videos)
- Frames = video seconds at 1 FPS (floored to an even count). The 768-frame
  cap applies to 1 video only (036, 779 s, 1 question).
- Frame sizes after qwen_omni_utils: 476x280 for 341 videos (135 video tokens
  per 2 frames); other aspect ratios for the rest (up to 165 tokens per 2
  frames for 4:3 videos at 476x364).
- Prompt tokens: min 4945, median 33399, max 62265 (of 65536). Video tokens up
  to 52800, audio tokens up to 10125. Three videos use more than 90% of the
  context.
- **1 video does not fit**: video 293 (692 s, 4:3, 480x360 -> 476x364) needs
  66475 tokens, 939 more than 65536. It has **4 questions** (qa 412, 457, 669
  L2; 790 L1). With the current policy they are recorded as
  `prompt_too_long` and score zero (0.4% of the benchmark).
- No other truncation or error. No video has a silent audio track. Audio and
  video durations differ by at most 0.49 s.
- Preprocessing (decord + audioread/librosa): median 2.5 s per video.

### (c) Response format (`qwen3omni-instr_av_greedy_check50`, 50 questions)
- 47/50 are a JSON object in a ```json fence (fences are accepted by both
  parsers).
- 3/50 hit `max_tokens=16384` (`finish_reason=length`), all Level 3: the
  model writes `"answer": "X"` correctly and then loops on evidence intervals
  past the end of the video (up to "61:52"). Classified `truncated_json`; the
  permissive letter is the real answer field.
- **Timestamp format**: 109 timestamps in total, 67 (61%) in the strict
  `[mm:ss,mm:ss]` format. The rest have **no square brackets**
  (`"03:51,03:52"`), are single points (`"00:14"`), or are zero-length
  (`"08:00,08:00"`).
  - Strict parsing fails on 17/50 questions (14 invalid timestamp, 3
    truncated JSON).
  - The authors' `parse_evidence_interval` also requires brackets, so
    bracket-less intervals give **no grounding credit in either mode**.
    That is why C.G./B.G. are the same in strict and permissive (11.0/4.2),
    while Acc. is 22 vs 30.
- Predicted intervals are short: median 2 s (GT median 5 s, up to 31 s), and
  usually one interval per modality, often the same for audio and visual.

## Decisions after step 2

- **Video 293** (692 s, 4:3, 66475 prompt tokens > 65536) is left as is. Its
  4 questions (qa 412, 457, 669 L2; 790 L1) are recorded as
  `status=error, error_type=prompt_too_long` and score zero in every scoring
  mode (0.4% of the benchmark; at most -0.4 points of overall Acc.). We do
  not lower its resolution or drop frames.
- **Video 036 lasts 779 s**, longer than the 1–12 minutes stated by the paper
  (§3.3, App. A.1). It is the only video above 768 s (the qwen_omni_utils
  frame cap; see "Long videos" above) and has 1 question.
- `max_tokens = 16384` (model card) is kept.

## Output length vs max_model_len

- vLLM 0.17.1 source (`v1/core/sched/utils.py::check_stop`): a request is
  finished when `num_tokens >= max_model_len` **or**
  `num_output_tokens >= max_tokens`. Both give `FINISHED_LENGTH_CAPPED`,
  i.e. `finish_reason="length"`. There is **no error**: the output is
  truncated. Only a prompt that is already longer than max_model_len is
  rejected at input validation (video 293).
- For the longest fitting video (036, 62265 prompt tokens) at most
  65536 − 62265 = 3271 output tokens are possible. A normal answer is about
  70 tokens, so this only matters for degenerate looping outputs.
- Every answer stores `stop_cause` (`natural_stop` / `max_tokens_reached` /
  `max_model_len_reached`, derived from vLLM's finish_reason, prompt length
  and output tokens), `n_output_tokens` and `prompt_tokens`. Runs made before
  this was added (check10, check50) have only `finish_reason` and
  `n_output_tokens`.
- Empirical check: `custom/tools/test_maxlen.py` (video 036, ignore_eos=True,
  so the model cannot stop before the context is full) -> result in
  `custom/output/_tests/maxlen_036.json`. **Not run yet**: on 2026-11-13 at
  16:00 the 4 GPUs were taken by another user's job (about 47 GB each, 100%
  utilisation), and our model needs about 76 GB per GPU.

## Scoring additions (after step 2)

- `diag_no_brackets` (**diagnostic only**, never a main number): permissive
  parsing, plus an interval written `mm:ss,mm:ss` without square brackets is
  accepted. Single points (`"00:14"`), zero-length intervals and
  end < start stay invalid. It shows how much C.G./B.G. are lost purely to the
  bracket format. RUNS.md has a separate column for it.
- `format_errors` in evaluation/results.json counts format problems:
  - per question: fully_well_formed, json_truncated, no_json_free_text,
    answer_not_single_letter, missing/empty evidence lists,
    generation_cut_*, has_<timestamp problem>;
  - per timestamp: strict_ok, missing_brackets, single_point, zero_length,
    end_before_start, readable_non_strict, other_unparseable,
    beyond_video_end (end > video duration + 1 s; syntactically valid, so it
    still counts in the authors' scorer).
  Example qa_ids are kept for each category.
- `stop_causes` counts in evaluation/results.json.
- `RUNS.md` is rebuilt from all run folders at every scoring, so it cannot
  drift from them.
- `score.py --run-dir` writes `evaluation/evaluation.txt`: Table 2 of the paper, every
  mode per level and overall with the difference ours − paper, 95% CI,
  baselines, and the main diagnostics. (It replaces the former
  `tools/compare_paper.py` / `paper_comparison.md`; `evaluation/results.json`
  replaces scores.json and input_stats_summary.json.)
- `custom/tools/compare_runs.py A B` recomputes Acc./C.G./B.G. on the
  questions the two runs have in common (e.g. sub200-r2 vs the same 200
  questions of the full run), per mode, and reports: letters changed (also by
  level), correctness flips, and the number and mean size of per-question
  changes in C.G./B.G. It only prints (no file).

## Running the long runs by hand: `custom/run_weekend.sh`

- The runs are launched by hand from tmux, on ONE GPU (`--gpu N` or `GPU=N`),
  never automatically. Order: max_model_len test -> greedy full -> greedy
  sub200-r2 -> (optional) sampling s1234 full -> (optional) sampling s4321
  sub200. Each run is followed by scoring and the comparisons.
- The script refuses to start a step if the GPU has less than 80000 MiB
  free (it does not wait and does not share), and refuses to start a run whose
  folder already exists (that would otherwise silently become `_v2`). For an
  interrupted run, use `--resume <folder>` as explained at the top of the file.
- `--dry-run` prints every command. No log file is written; redirect with
  `2>&1 | tee <file>` if needed.
- The runner now exits non-zero if a worker died or any question failed for a
  reason other than `prompt_too_long` (video 293), so `set -e` stops the
  script.
- Estimated time on 1 GPU (from check50: about 5 s per normal question; an
  output that loops until max_tokens takes about 6 min, at about 47 tokens/s
  with max_num_seqs=1; 3 of 8 Level-3 questions looped in check50):
  test about 5 min, greedy full about 4–8 h, sub200 about 1–2 h each,
  sampling full about 4–8 h. The uncertainty comes from how many answers loop
  to 16384 tokens.

## Constrained decoding experiment (`--constrained json`, 2026-11-16) — NOT part of the replication

Why: in the greedy full run about a third of the timestamps have no brackets
(`"03:50,03:52"`) and about 10% of the answers loop until max_tokens. The
authors' strict parser then discards the whole answer. The prompt already asks
for `"[mm:ss,mm:ss]"` twice, so changing the prompt would not help and would
depart from the paper. Instead, `--constrained json` keeps the prompt identical
and lets vLLM allow only the tokens that keep the output valid against a JSON
schema (`evidence_schema()` in `run_qwen_omni.py`).

- Enforced: a JSON object with `answer`, `visual_evidence` and `audio_evidence`
  in this order; `answer` is one of the question's option letters (A–H; qa 209:
  A–I; qa 483: A–F); `time` matches `^\[\d{2}:[0-5]\d,\d{2}:[0-5]\d\]$`;
  at most 100 intervals per list (user's choice: 100, not 10).
- Not enforced (left to the model): how many intervals (0 is allowed), end >
  start, times within the video. Whitespace and line breaks are free.
- Side effect: no Markdown fence is possible, so the output must start with `{`
  (the free model usually starts with ```` ```json ````).
- Backend: vLLM `auto` resolves to xgrammar 0.1.29 for this schema (checked with
  `has_xgrammar_unsupported_json_features`). Checked offline (CPU, Qwen
  tokenizer): the grammar accepts the prompt's format, compact JSON, empty
  lists and 100 intervals; it rejects no brackets, single points, seconds > 59,
  fences, other field orders, a letter outside the options, text after the
  letter, and 101 intervals.
- The `structured_output` block is added to `[settings]` only when the option
  is used, so the settings of unconstrained runs are unchanged (checked: they
  are still identical to those of the running `qwen3omni_av_greedy_sub100`,
  so `--resume` keeps working).
- Run: `bash custom/run_test_sub100_json.sh <gpu>` →
  `qwen3omni_av_greedy-json_sub100`, then compare with
  `qwen3omni_av_greedy_sub100` using `custom/tools/compare_runs.py`.

## max_tokens 2048 for free greedy runs (2026-11-17)

In `qwen3omni-instr_av_greedy_full` about 80% of the run time (≈4.5 of 5.5 h)
went into the 102 answers that loop until max_tokens (1.6M of 1.66M generated
tokens). All 102 are loops inside the evidence list; the longest answer that
stopped by itself has 338 tokens; none of the 102 has C.G. > 0 in any mode.
Greedy generation does not depend on max_tokens (it only stops earlier), so with
max_tokens 2048 the normal answers are unchanged and the loops score as before
(strict: 0; diagnostic: same letter, no evidence). The adapter default stays
16384 (model card); 2048 is passed explicitly by
`custom/run_full_greedy_maxtok2048.sh` → `qwen3omni_av_greedy_full_maxtok-2048`.
Not for sampling (one sampled answer stopped by itself after 12,746 tokens) and
not for constrained runs (answers with 100 intervals reach ≈4,200 tokens).

## Modality ablation: video only, audio only, text only (2026-11-17)

- `--input-mode v` (video frames, no audio), `a` (the video's audio track, no
  frames), `t` (question and options only). The missing input is **omitted**,
  no placeholder (no black frames, no silent audio: that variant is `av-mute`).
  The repository's prompts.py has a prompt for each input (`video_only`,
  `audio_only`, `text_only`) that tells the model what it has; we use it.
- Checked on CPU: `v` frames and `a` waveform are bit-identical to those of `av`
  (`a` uses the same `process_audio_info` call, without reading frames); the
  chat template puts `<|audio_pad|>` / `<|video_pad|>` / nothing before the text.
  The `[settings]` of `av` runs are unchanged (checked), so `--resume` still works.
- Scoring (`score.py`): the evidence lists the prompt does not ask for are scored
  as empty in every mode (strict: JSON re-validated with them set to []). So
  C.G. <= Acc/2 and B.G. = 0 by construction, as in the paper's Table 4
  (MiMo audio-only 30.0 / 10.0 / 0.0). evaluation.txt adds the grounding of the
  given modality alone. `av` scores unchanged (checked on 3 runs).
- Paper: Table 6(d) gives Qwen3-Omni-Instruct audio-only 18.2 and video-only
  12.5, accuracy only, counting a question as correct only if it is correct in
  BOTH of two independent runs (text-only there uses other models). Table 4 has
  no Qwen3-Omni. App. A.4: Qwen3-Omni itself was used to screen out questions
  answerable with one modality, so low unimodal scores are expected by design.
- Two runs per input (all lines in `custom/run_unimodal.sh`, after three 10-question tests),
  `custom/tools/both_runs_acc.py RUN_1 RUN_2` for the paper's "both" rule.
  On the two greedy av runs the rule gives 27.4 (diag) vs 29.2 / 28.5 per run.
- The paper does not say how the missing modality was handled (omitted or
  placeholder); omitting it is our choice, consistent with the prompts.

## Audio-evidence masking (2026-11-18)

- Question: does the model use the annotated audio evidence? For each question the
  audio evidence is replaced (not cut) on a copy of the 16 kHz waveform, just
  before the request (`custom/audio_mask.py`, `--audio-mask` in
  `run_qwen_omni.py`, av only). Length, timing, audio tokens, frames and prompt
  are as in av. The waveform from `preprocess` is shared by all questions of a
  video, hence the copy.
- Segments: each `evidence.audio` interval +/- 1 s (`--mask-pad`, annotations
  have 1 s resolution), clipped to the track, overlapping ones merged.
- Fill: `silence` (zeros) or `noise` (white Gaussian, RMS of the whole original
  track, seed from qa_id, clipped to [-1, 1]). 10 ms linear fade inside each
  segment; samples outside are untouched (self-test).
- Control (`--audio-mask random`): the same number of segments with the same
  lengths, uniformly placed (longest first) outside the padded evidence and not
  overlapping each other; seed from qa_id only, so positions are identical for
  silence and noise and across repeats. **7 questions** cannot fit them on their
  real track (qa 74, 376, 754, 770, 792, 816, 827; evidence 30–85% of the track):
  nothing is masked and the record says `control_infeasible: true`. Exclude
  them from evidence-vs-control comparisons.
- Each record has `input.audio_mask`: segments, masked seconds, noise RMS and the
  sha256 of the masked waveform. `custom/tools/viewer.py` (streamlit) rebuilds
  the masked audio with the same function and checks the hash, so what it plays
  is what the model received (after AAC encoding for the browser). Checked on
  video 179: viewer hash == run hash, for evidence/silence and random/noise.
- Runs: `custom/run_audiomask.sh`, 4 conditions x 2 repeats; reference: the two
  greedy av runs.

## Greedy run-to-run nondeterminism: investigation (2026-11-19)

Pairs of runs with identical settings, % answers identical character by character:
text-only t r1/r2 100%; av-maskev-noise r1/r2 100%; av-maskrand-noise r1/r2 100%;
av-maskev-silence 28%; av-maskrand-silence 28%; audio-only a 27%; video-only v 35%.
(Pairs on different subsets or max_tokens are not comparable: what ran before in the
process also differs.)

Ruled out, from the saved outputs and offline checks:
- different inputs: audio sha256, prompt/audio/video token counts identical in r1/r2;
- cache reuse between questions of a video: the FIRST question of each video differs
  as often as the others (silence: 101/391 identical);
- one process/GPU "unlucky": in the silence pairs both GPU shards differ (135/515,
  154/508), in the noise pairs both are 100%;
- CPU audio features (WhisperFeatureExtractor): bit-identical in 3 processes for
  original, silence and noise audio.
So the variation arises on the GPU inside vLLM and depends on the input; the
noise runs show the stack CAN be bit-reproducible across processes. Mechanism not
found in the vLLM 0.17.1 source.

Test: `custom/run_determinism.sh` (subset det48: 12 videos <= 5 min x 4 questions),
each configuration twice (-a/-b): base, noise, --no-prefix-caching,
VLLM_ENABLE_V1_MULTIPROCESSING=0, --batch-invariant (VLLM_BATCH_INVARIANT=1 +
FLASH_ATTN). Report: `custom/tools/determinism_report.py`. The new options add keys
to [settings] only when used (av settings checked unchanged).

### Determinism test results (det48, 2026-11-19)

Each configuration was run twice in separate processes (custom/run_determinism.sh; report: custom/tools/determinism_report.py). In every pair, settings and inputs (prompt tokens, audio hash) were identical.

| configuration | identical answers | same letter |
|---|---|---|
| base | 17/48 (35%) | 47/48 |
| noise (evidence -> noise) | 18/48 (38%) | 47/48 |
| noprefix | 15/48 (31%) | 46/48 |
| nomp (VLLM_ENABLE_V1_MULTIPROCESSING=0) | 19/48 (40%) | 48/48 |
| batchinv (VLLM_BATCH_INVARIANT=1, FLASH_ATTN) | 20/48 (42%) | 48/48 |

- **No configuration is deterministic, and on det48 the noise runs are not deterministic either.** The bit-identical full noise runs (2026-11-18) are not reproduced here; why remains unexplained.
- **Answers diverge early.** The median divergence point is about 68 characters, inside the first timestamp. For qa 1 (the first question of the run) they diverge at the first token (` ```json ` vs `{`), so even the prefill logits of the first request differ between processes.
- **Batch-invariant mode does not help**, so the cause is not batch composition or kernel choice in the language model.

Next step: custom/run_probe.sh with custom/tools/determinism_probe.py. It fingerprints every stage (CPU inputs, vision/audio encoder input and output, language-model input and output, first-token logprobs) in two processes, to find the first stage that differs. A second pair runs with enforce_eager.
