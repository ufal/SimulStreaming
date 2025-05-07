Code for Simultationeus LLM Translation with EuroLLM
====================================================

EuroLLM-9B-Instruct
- dir with huggingface model 
- created with `git clone https://huggingface.co/utter-project/EuroLLM-9B-Instruct`
- or with `git clone git@hf.co:utter-project/EuroLLM-9B-Instruct` -- after authorising to hf.co 


ct2_EuroLLM-9B-Instruct
- CTranslate2 converted model
- created with `ct2-transformers-converter --model EuroLLM-9B-Instruct/ --output_dir ct2_EuroLLM-9B-Instruct`


gold-asr-dir/
- ACL6060 dev set source, converted to the file format that the simul_llm_translate.py script uses on the source

sentence_segmenter.py
simul_llm_translate.py
- the code for EuroLLM inference with LocalAgreement

Dependencies: 
- TODO: installation instructions
- The dependencies are CTranslate2, with this

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lnet/work/people/machacek/smluvni-2024/alignatt-whisper.202412/sv-2/p3-fw/lib/python3.10/site-packages/nvidia/cublas/lib:/lnet/work/people/machacek/smluvni-2024/alignatt-whisper.202412/sv-2/p3-fw/lib/python3.10/site-packages/nvidia/cudnn/lib

- and transformers and tokenizer, probably

Usage:

- input can be either text file with timestamps in specific format

cat gold-asr-dir//2022.acl-long.110.txt | python3 simul_llm_translate.py --min-chunk-size 1 --language de --language-specific-len-threshold --max-context-length 80 --buffer_trimming sentences


- or *instance.log format of IWSLT2025 
python3 simul_llm_translate.py --input-instance gold-asr-dir//2022.acl-long.110.instance.log --min-chunk-size 1 --language de --max-context-length 300 | tee out

- I used the following parameters:

python3 simul_llm_translate.py --input-instance $input \
--min-chunk-size $ch \
--language $language \
--language-specific-len-threshold --buffer_trimming $trim \
--max-context-length $max_context_len

- the parameters should be explained in the help message, in the code, or in the paper


txt-to-instances.py
zh-ja-txt-to-instances-nobreaking+nospaces.py
- scripts that convert the default txt output of simul_llm_translate.py to instances
- the first works for En->De, the other for En->Zh and Ja
- because of a bug in Stream_LAAL.py, for Ja and Zh, spaces and newlines are removed from the outputs

Usage:

python3 zh-ja-txt-to-instances-nobreaking+nospaces.py < res/ja/asr.ch-1.4-frame-15-beam-1+eurollm.ch-4.unaware.gputype-any.i-1.model-eurollm-9b.language-ja/2022.acl-long.590.txt 2022.acl-long.590.wav > inst.log

slaal-ja-zh.sh
slaal-de.sh
- script that runs SLAAL for the whole dev set. Read how it works
- expects some dependency files (dev data) in correct format and paths. Adapt them for your worksystem

Usage: for one document only
./slaal-de.sh de-output/2022.acl-long.110.txt de-output/2022.acl-long.110.mw-segments > de-output/2022.acl-long.110.slaal

Usage: For the dir where the whole dev set is processed. It will create *instance.log for every document, and instances.log that concatenates them. And also mw-segments dir where the candidates are aligned with MWERSegmenter to the reference, and can be read by bare eyes (with a ziplines script and less, e.g.)

./slaal-de.sh de-output/ > de-output/slaal

- versioning: I have all this code in a private repo (quite messy at the moment). If you'd like to improve my code, or use my updates that I will do in coming weeks, I will add you to my repo. 





I could share more, if you need:
	- Update of Stream_LAAL.py that I use: 
	- the selected ASR candidates outputs, to evaluate realistic ASR quality and latency


Út 29. dubna 2025, 11:05:35 CEST
Dominik Macháček
