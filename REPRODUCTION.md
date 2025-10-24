# Reproduction Guide for SimulStreaming IWSLT 2025 Results

This guide provides instructions for reproducing the results from the CUNI submission to IWSLT 2025.

## Paper Information

**Title:** Simultaneous Translation with Offline Speech and LLM Models in CUNI Submission to IWSLT 2025

**Authors:** Dominik Macháček and Peter Polák (Charles University)

**Links:**
- arXiv: https://arxiv.org/abs/2506.17077
- ACL Anthology: https://aclanthology.org/2025.iwslt-1.41/
- GitHub: https://github.com/ufal/SimulStreaming

## Evaluation Dataset: ACL 60-60

The system was evaluated on the ACL 60-60 development set from IWSLT 2023.

### Dataset Details

- **Source:** ACL 2022 technical presentations (English)
- **Target Languages:** 10 languages (Arabic, Chinese, Dutch, French, German, Japanese, Farsi, Portuguese, Russian, Turkish)
- **Format:** Unsegmented WAV audio with gold sentence-segmented transcripts and translations
- **Download:** http://i13pc106.ira.uka.de/~jniehues/IWSLT-SLT/data/eval/en-xx/IWSLT-SLT.ACLdev2023.en-xx.tgz

### Dataset Structure

```
IWSLT-SLT.ACLdev2023.en-xx/
├── wav/                          # Full unsegmented audio files
├── segments/                     # SHAS baseline segmentation
├── gold/                         # Gold sentence-segmented transcripts/translations (XML)
└── README                        # Dataset documentation
```

## System Components

### 1. Speech-to-Text (SimulStreaming Whisper)

**Model:** Whisper large-v3
**Policy:** AlignAtt simultaneous policy
**Key Parameters:**
- `--frame_threshold`: 15-25 frames (controls latency/quality trade-off)
- `--beams`: 1 (greedy) or higher for beam search
- `--task`: transcribe or translate
- `--vac`: Enable Voice Activity Controller (recommended)

### 2. Text-to-Text Translation (Optional EuroLLM Cascade)

**Model:** EuroLLM-9B-Instruct
**Policy:** LocalAgreement
**Setup:** Requires CTranslate2 conversion

## Reproduction Steps

### Step 1: Install Dependencies

```bash
# Core dependencies
pip install -e ".[dev]"

# For translation cascade (optional)
pip install ctranslate2 transformers
```

### Step 2: Download ACL 60-60 Development Set

```bash
wget http://i13pc106.ira.uka.de/~jniehues/IWSLT-SLT/data/eval/en-xx/IWSLT-SLT.ACLdev2023.en-xx.tgz
tar -xzf IWSLT-SLT.ACLdev2023.en-xx.tgz
```

### Step 3: Download Models

**Whisper Model** (auto-downloaded on first run):
```bash
# large-v3 will be downloaded automatically to ./large-v3.pt
```

**EuroLLM Model** (for translation cascade):
```bash
# Clone from Hugging Face
git clone https://huggingface.co/utter-project/EuroLLM-9B-Instruct

# Convert to CTranslate2 format
ct2-transformers-converter \
    --model EuroLLM-9B-Instruct/ \
    --output_dir ct2_EuroLLM-9B-Instruct
```

### Step 4: Run Speech Translation

**English to German (direct Whisper):**
```bash
python3 simulstreaming_whisper.py \
    IWSLT-SLT.ACLdev2023.en-xx/wav/2022.acl-long.110.wav \
    --language en \
    --task translate \
    --frame_threshold 15 \
    --beams 1 \
    --vac \
    --comp_unaware \
    > output/2022.acl-long.110.de.txt
```

**English to German (with EuroLLM cascade):**
```bash
# 1. Generate English transcription
python3 simulstreaming_whisper.py \
    IWSLT-SLT.ACLdev2023.en-xx/wav/2022.acl-long.110.wav \
    --language en \
    --task transcribe \
    --frame_threshold 15 \
    --comp_unaware \
    > output/2022.acl-long.110.en.txt

# 2. Translate with EuroLLM
cd translate
python3 simul_llm_translate.py \
    --input-instance ../output/2022.acl-long.110.en.txt \
    --min-chunk-size 1 \
    --language de \
    --max-context-length 300 \
    > ../output/2022.acl-long.110.de.eurollm.txt
```

### Step 5: Evaluate Results

**Install Evaluation Tools:**
```bash
# Install mwerSegmenter (for automatic resegmentation)
# Install SacreBLEU
pip install sacrebleu
```

**Score with BLEU:**
```bash
# Resegment hypothesis to match reference format
mwerSegmenter \
    -mref IWSLT-SLT.ACLdev2023.en-xx/gold/2022.acl-long.110.de.xml \
    -hyp output/2022.acl-long.110.de.txt \
    -out output/2022.acl-long.110.de.segmented.txt

# Calculate BLEU
sacrebleu \
    IWSLT-SLT.ACLdev2023.en-xx/gold/2022.acl-long.110.de.xml \
    < output/2022.acl-long.110.de.segmented.txt
```

**Measure Latency (StreamLAAL):**
```bash
# For translation cascade, use SLAAL scripts in translate/
cd translate
./slaal-de.sh ../output/2022.acl-long.110.de.txt > ../output/2022.acl-long.110.slaal
```

## Key Configuration Parameters

### For High Quality (Lower Latency Tolerance)
```bash
--frame_threshold 25
--beams 3
--vac
--max_context_tokens 100
```

### For Low Latency (Higher Speed Priority)
```bash
--frame_threshold 15
--beams 1
--audio_min_len 0.5
```

### Language-Specific Settings

From the paper, the authors report:
- **Czech → English:** Direct Whisper translation
- **English → German:** EuroLLM cascade with `--min-chunk-size 1.4`, `--max-context-length 300`
- **English → Chinese/Japanese:** EuroLLM cascade with language-specific thresholds

## Expected Results

Based on the paper (IWSLT 2025 ACL 60-60 dev set):

### BLEU Scores
- **Czech → English:** Competitive with top systems
- **English → German:** 13-22 BLEU points (depending on configuration)
- **English → Chinese:** 13-22 BLEU points
- **English → Japanese:** 13-22 BLEU points

### Latency
- **Computation-aware:** Measured with actual processing time
- **Computation-unaware:** Theoretical lower bound (use `--comp_unaware` flag)
- Novel metric proposed: Enhanced measure of speech recognition latency

## Simulation Modes

1. **Computationally Aware** (default): Includes actual processing time
2. **Computationally Unaware** (`--comp_unaware`): Timer stops during computation
3. **Start-at** (`--start_at TIME`): Jump to specific time for debugging

## Troubleshooting

### Model Download Issues
If Whisper model download fails, manually download from:
https://openaipublic.azureedge.net/main/whisper/models/large-v3.pt

### CUDA Memory Issues
- Reduce `--beams` to 1
- Use `--audio_max_len 25.0` (default 30.0)
- Ensure GPU has sufficient memory (8GB+ recommended)

### Triton Fallback
System gracefully falls back to CPU implementations if CUDA/Triton fails.
Check warnings in logs - performance may be slower but functionality preserved.

## Citation

If you use SimulStreaming or reproduce these results, please cite:

```bibtex
@inproceedings{machacek-polak-2025-simultaneous,
    title = "Simultaneous Translation with Offline Speech and {LLM} Models in {CUNI} Submission to {IWSLT} 2025",
    author = "Macháček, Dominik and Polák, Peter",
    booktitle = "Proceedings of the 22nd International Conference on Spoken Language Translation",
    year = "2025",
    url = "https://aclanthology.org/2025.iwslt-1.41",
    pages = "412--481"
}
```

## Additional Resources

- **IWSLT 2023 Multilingual Track:** https://iwslt.org/2023/multilingual
- **ACL 60-60 Paper:** https://aclanthology.org/2023.iwslt-1.2/
- **Evaluation Scripts:** Contact authors or check IWSLT organizers' repository
- **EuroLLM Model:** https://huggingface.co/utter-project/EuroLLM-9B-Instruct

## Support

For questions or issues:
- GitHub Issues: https://github.com/ufal/SimulStreaming/issues
- Contact: Dominik Macháček <machacek@ufal.mff.cuni.cz>
