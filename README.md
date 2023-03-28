# Awesome-Healthcare-Foundation-Models 

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

Curated list of awesome large AI models (LAMs), or foundation models, in healthcare. We organize the current LAMs into four categories: large language models (LLMs), large vision models (LVMs), large audio models (LAudiMs), and large multi-modal models (LMMs). The areas that these LAMs are applied to include but not limited to bioinformatics, medical diagnosis and decision making, medical imaging and vision, medical informatics, medical education, public health, and medical robotics.

We welcome contributions to this repository to add more resources. Please submit a pull request if you want to contribute!

## Survey

This repository is largely based on the following paper:

**[Large AI Models in Health Informatics:
Applications, Challenges, and the Future](https://arxiv.org/pdf/2303.11568v1.pdf)**
<br />
by 
Jianing Qiu,
Lin Li, 
Jiankai Sun,
Peilun Shi,
Ruiyang Zhang,
Yinzhao Dong,
Kyle Lam,
Frank P.-W. Lo,
Bo Xiao,
Wu Yuan,
Dong Xu, and
Benny Lo
<br />


If you find this repository helpful, please consider citing:

```bibtex
@article{qiu2023large,
  title={Large AI Models in Health Informatics: Applications, Challenges, and the Future},
  author={Qiu, Jianing and Li, Lin and Sun, Jiankai and Peng, Jiachuan and Shi, Peilun and Zhang, Ruiyang and Dong, Yinzhao and Lam, Kyle and Lo, Frank P-W and Xiao, Bo and others},
  journal={arXiv preprint arXiv:2303.11568},
  year={2023}
}
```

## Large Language Models

* ChatCAD: Interactive Computer-Aided Diagnosis on Medical Image using
Large Language Models [[Paper]](https://arxiv.org/pdf/2302.07257.pdf)

* DeID-GPT: Zero-shot Medical Text De-Identification by GPT-4 [[Paper]](https://arxiv.org/pdf/2303.11032.pdf)

* Capabilities of GPT-4 on Medical Challenge Problems [[Paper]](https://arxiv.org/pdf/2303.13375.pdf)

* BioBERT: a pre-trained biomedical language representation model for biomedical text mining [[Paper]](https://arxiv.org/pdf/1901.08746.pdf)

* Publicly Available Clinical BERT Embeddings [[Paper]](https://arxiv.org/pdf/1904.03323.pdf)

* BioMegatron: Larger Biomedical Domain Language Model [[Paper]](https://arxiv.org/pdf/2010.06060.pdf)

* Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks [[Paper]](https://aclanthology.org/2020.acl-main.740.pdf)

* Med-BERT: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction [[Paper]](https://www.nature.com/articles/s41746-021-00455-y)

* BioELECTRA:Pretrained Biomedical text Encoder using Discriminators [[Paper]](https://aclanthology.org/2021.bionlp-1.16.pdf)

* LinkBERT: Pretraining Language Models with Document Links [[Paper]](https://arxiv.org/pdf/2203.15827.pdf)

* BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining [[Paper]](https://arxiv.org/pdf/2210.10341.pdf)

* Large Language Models Encode Clinical Knowledge [[Paper]](https://arxiv.org/pdf/2212.13138.pdf)

* A large language model for electronic health records [[Paper]](https://www.nature.com/articles/s41746-022-00742-2)

* Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing [[Paper]](https://arxiv.org/pdf/2007.15779.pdf)

* BEHRT: Transformer for Electronic Health Records [[Paper]](https://www.nature.com/articles/s41598-020-62922-y)







## Large Vision Models






## Large Audio Models
* wav2vec: Unsupervised Pre-training for Speech Recognition [[Paper]](https://arxiv.org/abs/1904.05862) [[Blog]](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/)
* W2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training [[Paper]](https://arxiv.org/abs/2108.06209)
* AudioLM: a Language Modeling Approach to Audio Generation [[Paper]](https://arxiv.org/abs/2209.03143) [[Project]](https://google-research.github.io/seanet/audiolm/examples/) [[Blog]](https://ai.googleblog.com/2022/10/audiolm-language-modeling-approach-to.html)
* HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units [[Paper]](https://arxiv.org/abs/2106.07447) [[HuggingFace]](https://huggingface.co/docs/transformers/model_doc/hubert)
* XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale [[Paper]](https://arxiv.org/abs/2111.09296) [[Blog]](https://ai.facebook.com/blog/xls-r-self-supervised-speech-processing-for-128-languages/) [[HuggingFace]](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
* MusicLM: Generating Music From Text [[Paper]](https://arxiv.org/abs/2301.11325) [[Project]](https://google-research.github.io/seanet/musiclm/examples/) [[Code]](https://github.com/lucidrains/musiclm-pytorch)
* Diffsound: Discrete Diffusion Model for Text-to-sound Generation [[Paper]](https://arxiv.org/abs/2207.09983) [[Project]](http://dongchaoyang.top/text-to-sound-synthesis-demo/) [[Code]](https://github.com/yangdongchao/Text-to-sound-Synthesis)
* AudioGen: Textually Guided Audio Generation [[Paper]](https://arxiv.org/abs/2209.15352) [[Project]](https://felixkreuk.github.io/audiogen/)
* Whisper: Robust Speech Recognition via Large-Scale Weak Supervision [[Paper]](https://arxiv.org/abs/2212.04356) [[Code]](https://github.com/openai/whisper) [[HuggingFace]](https://huggingface.co/openai/whisper-tiny.en)
* Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages [[Paper]](https://arxiv.org/abs/2303.01037) [[Blog]](https://ai.googleblog.com/2023/03/universal-speech-model-usm-state-of-art.html)


## Large Multi-modal Models

* GPT-4 Technical Report [[Paper]](https://arxiv.org/pdf/2303.08774.pdf)

* Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning [[Paper]](https://www.nature.com/articles/s41551-022-00936-9)

* Contrastive Learning of Medical Visual Representations from Paired Images and Text [[Paper]](https://arxiv.org/pdf/2010.00747.pdf)

## Applications of Large AI Models in Health Informatics

###  Medical Robotics

* Decision Transformer: Reinforcement Learning via Sequence Modeling [[Paper]](https://arxiv.org/abs/2106.01345) [[Code]](https://github.com/kzl/decision-transformer)
* R3M: A Universal Visual Representation for Robot Manipulation [[Paper]](https://arxiv.org/abs/2203.12601) [[Project]](https://sites.google.com/view/robot-r3m/) [[Code]](https://github.com/facebookresearch/r3m)
* MimicPlay: Long-Horizon Imitation Learning by Watching Human Play [[Paper]](https://arxiv.org/abs/2302.12422) [[Project]](https://mimic-play.github.io/)
* PaLM-E: An Embodied Multimodal Language Model [[Paper]](https://arxiv.org/abs/2303.03378) [[Project]](https://palm-e.github.io/) [[Blog]](https://ai.googleblog.com/2023/03/palm-e-embodied-multimodal-language.html)
* A Generalist Agent [[Paper]](https://arxiv.org/abs/2205.06175) [[Blog]](https://www.deepmind.com/blog/a-generalist-agent)
* CLIPort: What and Where Pathways for Robotic Manipulation [[Paper]](https://arxiv.org/abs/2109.12098) [[Project]](https://cliport.github.io/) [[Code]](https://github.com/cliport/cliport)
* Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation [[Paper]](https://arxiv.org/abs/2209.05451) [[Project]](https://peract.github.io/) [[Code]](https://github.com/peract/peract)
* Do As I Can, Not As I Say: Grounding Language in Robotic Affordances [[Paper]](https://arxiv.org/abs/2204.01691) [[Project]](https://say-can.github.io/) [[Code]](https://github.com/google-research/google-research/tree/master/saycan)
* VIMA: General Robot Manipulation with Multimodal Prompts [[Paper]](https://arxiv.org/abs/2210.03094) [[Project]](https://vimalabs.github.io/) [[Code]](https://github.com/vimalabs/VIMA)
* RT-1: Robotics Transformer for Real-World Control at Scale [[Paper]](https://arxiv.org/abs/2212.06817) [[Project]](https://robotics-transformer.github.io/) [[Code]](https://github.com/google-research/robotics_transformer)
* ChatGPT for Robotics: Design Principles and Model Abilities [[Paper]](https://www.microsoft.com/en-us/research/uploads/prod/2023/02/ChatGPT___Robotics.pdf) [[Blog]](https://www.microsoft.com/en-us/research/group/autonomous-systems-group-robotics/articles/chatgpt-for-robotics/) [[Code]](https://github.com/microsoft/PromptCraft-Robotics)
