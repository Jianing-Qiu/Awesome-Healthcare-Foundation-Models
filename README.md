# Awesome-Healthcare-Foundation-Models

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

Curated list of awesome large AI models (LAMs), or foundation models, in healthcare. We organize the current LAMs into four categories: large language models (LLMs), large vision models (LVMs), large audio models, and large multi-modal models (LMMs). The areas that these LAMs are applied to include but not limited to bioinformatics, medical diagnosis, medical imaging, medical informatics, medical education, public health, and medical robotics.

We welcome contributions to this repository to add more resources. Please submit a pull request if you want to contribute!

## News

We are excited to annouce a _IEEE J-BHI_ special issue on **Biomedical and Health Foundation Models**. Please refer to the [call-for-papers](https://www.embs.org/jbhi/wp-content/uploads/sites/18/2023/06/JBHI_Foundation_Models_Call-for-Papers.pdf) for more details.

Topics of interest include but not limited to:

1. Basic research on new theories, principles, and structures of biomedical and health foundation models
2. Basic research on the interpretability and explainability of biomedical and health foundation models
3. Prompt engineering in biomedical and health foundation models
4. Data engineering in biomedical and health foundation models
5. Large-scale biomedical and health dataset
6. Multi-modal learning and alignment for biomedical and health foundation models
7. Efficient computing for biomedical and health foundation models
8. Adversarial robustness of biomedical and health foundation models
9. Applications of foundation models in biomedical and health informatics
10. New evaluation paradigms for biomedical and health foundation models
11. New computer systems for biomedical and health foundation models
12. Decentralised methods for developing and deploying biomedical and health foundation models
13. Foundation model ethics, safety, privacy, and regulations in biomedicine and healthcare

Please help spread the word and contribute if you are interested or already working on these topics!

## Table of Contents

- [Awesome-Healthcare-Foundation-Models](#awesome-healthcare-foundation-models)
  - [News](#news)
  - [Table of Contents](#table-of-contents)
  - [Survey](#survey)
  - [Large Language Models](#large-language-models)
    - [Healthcare Domain](#healthcare-domain)
    - [General Domain](#general-domain)
  - [Large Vision Models](#large-vision-models)
    - [Healthcare Domain](#healthcare-domain-1)
    - [General Domain](#general-domain-1)
  - [Large Audio Models](#large-audio-models)
    - [Healthcare Domain](#healthcare-domain-2)
    - [General Domain](#general-domain-2)
  - [Large Multi-modal Models](#large-multi-modal-models)
    - [Healthcare Domain](#healthcare-domain-3)
    - [General Domain](#general-domain-3)
  - [Applications of Large AI Models in Healthcare](#applications-of-large-ai-models-in-healthcare)
    - [Bioinformatics](#bioinformatics)
    - [Medical Diagnosis](#medical-diagnosis)
    - [Medical Imaging](#medical-imaging)
    - [Medical Informatics](#medical-informatics)
    - [Medical Education](#medical-education)
    - [Public Health](#public-health)
    - [Medical Robotics](#medical-robotics)
  - [AI Legislation](#ai-legislation)
  - [Large-scale Datasets in Biomedical and Health Informatics](#large-scale-datasets-in-biomedical-and-health-informatics)
    - [Open Source](#open-source)
    - [Private or Upon Approval](#private-or-upon-approval)

## Survey

This repository is largely based on the following paper:

**[Large AI Models in Health Informatics:
Applications, Challenges, and the Future](https://ieeexplore.ieee.org/document/10261199)**
<br />
Jianing Qiu,
Lin Li,
Jiankai Sun,
Jiachuan Peng,
Peilun Shi,
Ruiyang Zhang,
Yinzhao Dong,
Kyle Lam,
Frank P.-W. Lo,
Bo Xiao,
Wu Yuan,
Ningli Wang,
Dong Xu, and
Benny Lo
<br />

If you find this repository helpful, please consider citing:

```bibtex
@article{qiu2023large,
  title={Large ai models in health informatics: Applications, challenges, and the future},
  author={Qiu, Jianing and Li, Lin and Sun, Jiankai and Peng, Jiachuan and Shi, Peilun and Zhang, Ruiyang and Dong, Yinzhao and Lam, Kyle and Lo, Frank P-W and Xiao, Bo and others},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2023},
  publisher={IEEE}
}
```

![](LAM_key_feature_summary.png)

## Large Language Models

### Healthcare Domain

- ChiMed-GPT: A Chinese Medical Large Language Model with Full
Training Regime and Better Alignment to Human Preferences [[Paper]](https://arxiv.org/pdf/2311.06025.pdf) [[Code]](https://github.com/synlp/ChiMed-GPT)
- Med-PaLM 2: Towards Expert-Level Medical Question Answering with Large Language Models [[Paper]](https://arxiv.org/pdf/2305.09617.pdf)
- KeBioLM: Improving Biomedical Pretrained Language Models with Knowledge [[Paper]](https://arxiv.org/abs/2104.10344)
- BioELMo: Probing Biomedical Embeddings from Language Models [[Paper]](https://arxiv.org/abs/1904.02181)
- BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model [[Paper]](https://aclanthology.org/2022.bionlp-1.9.pdf)
- ClinicalT5: A Generative Language Model for Clinical Text [[Paper]](https://aclanthology.org/2022.findings-emnlp.398.pdf)
- GatorTron: A Large Clinical Language Model to Unlock Patient Information from Unstructured Electronic Health Records [[Paper]](https://arxiv.org/pdf/2203.03540v2.pdf)
- ChatCAD: Interactive Computer-Aided Diagnosis on Medical Image using Large Language Models [[Paper]](https://arxiv.org/pdf/2302.07257.pdf) [[Code]](https://github.com/zhaozh10/ChatCAD)
- DeID-GPT: Zero-shot Medical Text De-Identification by GPT-4 [[Paper]](https://arxiv.org/pdf/2303.11032.pdf)
- Capabilities of GPT-4 on Medical Challenge Problems [[Paper]](https://arxiv.org/pdf/2303.13375.pdf)
- BioBERT: a pre-trained biomedical language representation model for biomedical text mining [[Paper]](https://arxiv.org/pdf/1901.08746.pdf)
- Publicly Available Clinical BERT Embeddings [[Paper]](https://arxiv.org/pdf/1904.03323.pdf)
- BioMegatron: Larger Biomedical Domain Language Model [[Paper]](https://arxiv.org/pdf/2010.06060.pdf)
- Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks [[Paper]](https://aclanthology.org/2020.acl-main.740.pdf)
- Med-BERT: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction [[Paper]](https://www.nature.com/articles/s41746-021-00455-y)
- CPLLM: Clinical Prediction with Large Language Models [[Paper]](https://arxiv.org/abs/2309.11295) [[Code]](https://github.com/nadavlab/CPLLM)
- DoctorGLM: Fine-tuning your chinese doctor is not a herculean task [[Paper]](https://arxiv.org/abs/2304.01097) [[Code]](https://github.com/xionghonglin/DoctorGLM)
- HuatuoGPT, Towards Taming Language Models To Be a Doctor [[Paper]](https://arxiv.org/abs/2305.15075) [[Code]](https://github.com/FreedomIntelligence/HuatuoGPT)
- BioELECTRA:Pretrained Biomedical text Encoder using Discriminators [[Paper]](https://aclanthology.org/2021.bionlp-1.16.pdf)
- LinkBERT: Pretraining Language Models with Document Links [[Paper]](https://arxiv.org/pdf/2203.15827.pdf)
- BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining [[Paper]](https://arxiv.org/pdf/2210.10341.pdf)
- Large Language Models Encode Clinical Knowledge [[Paper]](https://arxiv.org/pdf/2212.13138.pdf)
- A large language model for electronic health records [[Paper]](https://www.nature.com/articles/s41746-022-00742-2)
- Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing [[Paper]](https://arxiv.org/pdf/2007.15779.pdf)
- BEHRT: Transformer for Electronic Health Records [[Paper]](https://www.nature.com/articles/s41598-020-62922-y)
- Federated Learning of Medical Concepts Embedding using BEHRT [[Paper]](https://arxiv.org/abs/2305.13052) [[Code]](https://github.com/nadavlab/FederatedBEHRT)
- RadBERT: Adapting Transformer-based Language Models to Radiology [[paper]](https://pubs.rsna.org/doi/epdf/10.1148/ryai.210258) [[HuggingFace]](https://huggingface.co/UCSD-VA-health/RadBERT-RoBERTa-4m)
- Highly accurate protein structure prediction with AlphaFold [[Paper]](https://www.nature.com/articles/s41586-021-03819-2) [[Code]](https://github.com/deepmind/alphafold)
- Accurate prediction of protein structures and interactions using a three-track neural network [[Paper]](https://www.science.org/doi/full/10.1126/science.abj8754?casa_token=tleEHPOOSr8AAAAA%3AT0eToIMPW0oN1jjIGLs8aPyQK8qbcFIByjT1x4k90tvBAj03SZUzpEinCPe_t-g4ECmjJ9wlj8OwQBs)
- Protein complex prediction with AlphaFold-Multimer [[Paper]](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.abstract)
- FastFold: Reducing AlphaFold Training Time from 11 Days to 67 Hours [[Paper]](https://arxiv.org/abs/2203.00854) [[Code]](https://github.com/hpcaitech/fastfold)
- HelixFold: An Efficient Implementation of AlphaFold2 using PaddlePaddle [[Paper]](https://arxiv.org/abs/2207.05477) [[Code]](https://github.com/PaddlePaddle/PaddleHelix)
- Uni-Fold: An Open-Source Platform for Developing Protein Folding Models beyond AlphaFold [[Paper]](https://www.biorxiv.org/content/10.1101/2022.08.04.502811v3.abstract) [[Code]](https://github.com/dptech-corp/Uni-Fold)
- OpenFold: Retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization [[Paper]](https://www.biorxiv.org/content/10.1101/2022.11.20.517210v2.abstract) [[Code]](https://github.com/aqlaboratory/openfold)
- ManyFold: an efficient and flexible library for training and validating protein folding models [[Paper]](https://academic.oup.com/bioinformatics/article/39/1/btac773/6887136) [[Code]](https://github.com/instadeepai/manyfold)
- ColabFold: making protein folding accessible to all [[Paper]](https://www.nature.com/articles/s41592-022-01488-1) [[Code]](https://github.com/sokrypton/ColabFold)
- Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences [[Paper]](https://www.pnas.org/doi/abs/10.1073/pnas.2016239118) [[Code]](https://github.com/facebookresearch/esm)
- ProGen: Language Modeling for Protein Generation [[Paper]](https://arxiv.org/abs/2004.03497) [[Code]](https://github.com/lucidrains/progen)
- ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing [[Paper]](https://arxiv.org/abs/2007.06225) [[Code]](https://github.com/agemagician/ProtTrans)
- Evolutionary-scale prediction of atomic level protein structure with a language model [[Paper]](https://www.science.org/doi/full/10.1126/science.ade2574)
- High-resolution de novo structure prediction from primary sequence [[Paper]](https://www.biorxiv.org/content/10.1101/2022.07.21.500999v1.abstract) [[Code]](https://github.com/HeliXonProtein/OmegaFold)
- Single-sequence protein structure prediction using a language model and deep learning [[Paper]](https://www.nature.com/articles/s41587-022-01432-w)
- Improved the Protein Complex Prediction with Protein Language Models [[Paper]](https://www.biorxiv.org/content/10.1101/2022.09.15.508065v2.abstract)
- MSA Transformer [[Paper]](http://proceedings.mlr.press/v139/rao21a.html) [[Code]](https://github.com/The-AI-Summer/self-attention-cv)
- Deciphering antibody affinity maturation with language models and weakly supervised learning [[Paper]](https://arxiv.org/abs/2112.07782)
- xTrimoABFold: De novo Antibody Structure Prediction without MSA [[Paper]](https://arxiv.org/abs/2212.00735)
- scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data [[Paper]](https://arxiv.org/abs/2212.00735) [[Code]](https://github.com/TencentAILabHealthcare/scBERT)
- Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions [[Paper]](https://www.biorxiv.org/content/10.1101/2022.08.06.503062v2.abstract) [[Code]](https://github.com/ml4bio/rna-fm)
- E2Efold-3D: End-to-End Deep Learning Method for accurate de novo RNA 3D Structure Prediction [[Paper]](https://www.biorxiv.org/content/10.1101/2022.08.06.503062v2.abstract) [[Code]](https://github.com/ml4bio/rna-fm)
- HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution [[Paper]](https://arxiv.org/abs/2306.15794) [[Code]](https://github.com/HazyResearch/hyena-dna)

### General Domain

- Chatgpt: Optimizing language models for dialogue [[Blog]](https://openai.com/blog/chatgpt/)
- LLaMA: Open and Efficient Foundation Language Models [[Paper]](https://arxiv.org/pdf/2302.13971.pdf)
- Scaling Instruction-Finetuned Language Models [[Paper]](https://arxiv.org/pdf/2210.11416.pdf)
- PaLM: Scaling Language Modeling with Pathways [[Paper]](https://arxiv.org/pdf/2204.02311.pdf)
- Training Compute-Optimal Large Language Models [[Paper]](https://arxiv.org/pdf/2203.15556.pdf)
- Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model [[Paper]](https://arxiv.org/pdf/2201.11990.pdf)
- BLOOM: A 176B-Parameter Open-Access Multilingual Language Model [[Paper]](https://arxiv.org/pdf/2211.05100.pdf)
- LaMDA: Language Models for Dialog Applications [[Paper]](https://arxiv.org/pdf/2201.08239.pdf)
- OPT: Open Pre-trained Transformer Language Models [[Paper]](https://arxiv.org/pdf/2205.01068.pdf)
- Training language models to follow instructions with human feedback [[Paper]](https://arxiv.org/pdf/2203.02155.pdf)
- Scaling Language Models: Methods, Analysis & Insights from Training Gopher [[Paper]](https://arxiv.org/pdf/2112.11446.pdf)
- Multitask prompted training enables zero-shot task generalization [[Paper]](https://arxiv.org/pdf/2110.08207.pdf)
- Language Models are Few-Shot Learners [[Paper]](https://arxiv.org/pdf/2005.14165.pdf)
- Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer [[Paper]](https://arxiv.org/pdf/1910.10683.pdf)
- RoBERTa: A Robustly Optimized BERT Pretraining Approach [[Paper]](https://arxiv.org/pdf/1907.11692.pdf)
- Language Models are Unsupervised Multitask Learners [[Paper]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Improving language models by retrieving from trillions of tokens [[Paper]](https://arxiv.org/pdf/2112.04426.pdf)
- WebGPT: Browser-assisted question-answering with human feedback [[Paper]](https://arxiv.org/pdf/2112.09332.pdf)
- Improving alignment of dialogue agents via targeted human judgements [[Paper]](https://arxiv.org/pdf/2209.14375.pdf)
- Improving Language Understanding by Generative Pre-Training [[Paper]](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [[Paper]](https://arxiv.org/pdf/1810.04805.pdf)

## Large Vision Models

### Healthcare Domain

- VisionFM: A Multi-Modal Multi-Task Vision Foundation Model for Generalist Ophthalmic Artificial Intelligence [[Paper]](https://arxiv.org/abs/2310.04992)
- RETFound: A foundation model for generalizable disease detection from retinal images [[Paper]](https://www.nature.com/articles/s41586-023-06555-x)
- EndoFM: Foundation Model for Endoscopy Video Analysis via Large-scale Self-supervised Pre-train [[Paper]](https://arxiv.org/abs/2306.16741) [[Code]](https://github.com/med-air/Endo-FM)
- STU-Net: Scalable and Transferable Medical Image Segmentation Models Empowered by Large-Scale Supervised Pre-training [[Paper]](https://arxiv.org/abs/2304.06716)
- LVM-Med: Learning Large-Scale Self-Supervised Vision Models for Medical Imaging via Second-order Graph Matching [[Paper]](https://arxiv.org/abs/2306.11925) [[Code]](https://github.com/duyhominhnguyen/LVM-Med)
- Med3d: Transfer learning for 3d medical image analysis [[Paper]](https://arxiv.org/abs/1904.00625) [[Code]](https://github.com/Tencent/MedicalNet)
- Models genesis: Generic autodidactic models for 3d medical image analysis [[Paper]](https://arxiv.org/abs/1908.06912) [[Code]](https://github.com/MrGiovanni/ModelsGenesis)
- MICLe: Big self-supervised models advance medical image classifications [[Paper]](https://arxiv.org/abs/2101.05224) [[Code]](https://github.com/rjrobben/MICLe_pytorch)
- C2l: Comparing to Learn: Surpassing ImageNet Pretraining on Radiographs By Comparing Image Representations [[Paper]](https://arxiv.org/abs/2007.07423) [[Code]](https://github.com/funnyzhou/C2L_MICCAI2020)
- MoCo-CXR: MoCo Pretraining Improves Representation and Transferability of Chest X-ray Models [[Paper]](https://arxiv.org/abs/2010.05352) [[Code]](https://github.com/stanfordmlgroup/MoCo-CXR)
- Transunet: Transformers make strong encoders for medical image segmentation [[Paper]](https://arxiv.org/abs/2102.04306) [[Code]](https://github.com/Beckschen/TransUNet)
- Transfuse: Fusing transformers and cnns for medical image segmentation [[Paper]](https://arxiv.org/abs/2102.08005) [[Code]](https://github.com/Rayicer/TransFuse)
- Medical transformer: Gated axial-attention for medical image segmentation [[Paper]](https://arxiv.org/abs/2102.10662) [[Code]](https://github.com/jeya-maria-jose/Medical-Transformer)
- UNETR: Transformers for 3D Medical Image Segmentation [[Paper]](https://arxiv.org/abs/2103.10504) [[Code]](https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV)
- Cotr: Efficiently bridging cnn and transformer for 3d medical image segmentation [[Paper]](https://arxiv.org/abs/2103.03024) [[Code]](https://github.com/YtongXie/CoTr)
- Swin-unet: Unet-like pure transformer for medical image segmentation [[Paper]](https://arxiv.org/abs/2105.05537) [[Code]](https://github.com/HuCaoFighting/Swin-Unet)
- SAM4Med: Generalist Vision Foundation Models for Medical Imaging: A Case Study of Segment Anything Model on Zero-Shot Medical Segmentation [[Paper]](https://arxiv.org/pdf/2304.12637.pdf)
- Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures[[Paper]](https://arxiv.org/abs/2307.15220) [[Code]](https://github.com/CAMMA-public/SurgVLP)

### General Domain

**CNNs**:

- GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism [[paper]](https://proceedings.neurips.cc/paper/2019/hash/093f65e080a295f8076b1c5722a46aa2-Abstract.html)
- Big Transfer (BiT): General Visual Representation Learning [[paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500477.pdf)
- Designing Network Design Spaces [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Radosavovic_Designing_Network_Design_Spaces_CVPR_2020_paper.html)
- Self-supervised Pretraining of Visual Features in the Wild [[paper]](http://arxiv.org/abs/2103.01988)
- EfficientNetV2: Smaller Models and Faster Training [[paper]](https://proceedings.mlr.press/v139/tan21a.html)
- A ConvNet for the 2020s [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf)
- InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions [[paper]](http://arxiv.org/abs/2211.05778)

**Vision Transformers**:

- Generative Pretraining From Pixels [[paper]](https://proceedings.mlr.press/v119/chen20s.html)
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale [[paper]](https://openreview.net/forum?id=YicbFdNTTy&utm_campaign=f86497ed3a-EMAIL_CAMPAIGN_2019_04_24_03_18_COPY_01&utm_medium=email&utm_source=Deep%20Learning%20Weekly&utm_term=0_384567b42d-f86497ed3a-72965345)
- Transformer in Transformer [[paper]](https://proceedings.neurips.cc/paper/2021/hash/854d9fca60b4bd07f9bb215d59ef5561-Abstract.html)
- Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows [[paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html)
- Training data-efficient image transformers & distillation through attention [[paper]](https://proceedings.mlr.press/v139/touvron21a.html)
- Self-supervised Models are Good Teaching Assistants for Vision Transformers [[paper]](https://proceedings.mlr.press/v162/wu22c.html)
- Scaling Vision with Sparse Mixture of Experts [[paper]](https://proceedings.neurips.cc/paper/2021/hash/48237d9f2dea8c74c2a72126cf63d933-Abstract.html)
- Going Deeper With Image Transformers [[paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.html)
- Masked Autoencoders Are Scalable Vision Learners [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html)
- Swin Transformer V2: Scaling Up Capacity and Resolution [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.html)
- Scaling Vision Transformers [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Zhai_Scaling_Vision_Transformers_CVPR_2022_paper.html)
- Efficient Self-supervised Vision Transformers for Representation Learning [[paper]](https://openreview.net/forum?id=fVu3o-YUGQK)
- Scaling Vision Transformers to 22 Billion Parameters [[paper]](http://arxiv.org/abs/2302.05442)

**CNNs + ViTs**:

- CoAtNet: Marrying Convolution and Attention for All Data Sizes [[paper]](https://proceedings.neurips.cc/paper/2021/hash/20568692db622456cc42a2e853ca21f8-Abstract.html)
- LeViT: A Vision Transformer in ConvNet's Clothing for Faster Inference [[paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Graham_LeViT_A_Vision_Transformer_in_ConvNets_Clothing_for_Faster_Inference_ICCV_2021_paper.html)
- ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases [[paper]](https://proceedings.mlr.press/v139/d-ascoli21a.html)

## Large Audio Models

### Healthcare Domain

### General Domain

- wav2vec: Unsupervised Pre-training for Speech Recognition [[Paper]](https://arxiv.org/abs/1904.05862) [[Blog]](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/)
- W2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training [[Paper]](https://arxiv.org/abs/2108.06209)
- AudioLM: a Language Modeling Approach to Audio Generation [[Paper]](https://arxiv.org/abs/2209.03143) [[Project]](https://google-research.github.io/seanet/audiolm/examples/) [[Blog]](https://ai.googleblog.com/2022/10/audiolm-language-modeling-approach-to.html)
- HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units [[Paper]](https://arxiv.org/abs/2106.07447) [[HuggingFace]](https://huggingface.co/docs/transformers/model_doc/hubert)
- XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale [[Paper]](https://arxiv.org/abs/2111.09296) [[Blog]](https://ai.facebook.com/blog/xls-r-self-supervised-speech-processing-for-128-languages/) [[HuggingFace]](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
- MusicLM: Generating Music From Text [[Paper]](https://arxiv.org/abs/2301.11325) [[Project]](https://google-research.github.io/seanet/musiclm/examples/) [[Code]](https://github.com/lucidrains/musiclm-pytorch)
- Diffsound: Discrete Diffusion Model for Text-to-sound Generation [[Paper]](https://arxiv.org/abs/2207.09983) [[Project]](http://dongchaoyang.top/text-to-sound-synthesis-demo/) [[Code]](https://github.com/yangdongchao/Text-to-sound-Synthesis)
- AudioGen: Textually Guided Audio Generation [[Paper]](https://arxiv.org/abs/2209.15352) [[Project]](https://felixkreuk.github.io/audiogen/)
- Whisper: Robust Speech Recognition via Large-Scale Weak Supervision [[Paper]](https://arxiv.org/abs/2212.04356) [[Code]](https://github.com/openai/whisper) [[HuggingFace]](https://huggingface.co/openai/whisper-tiny.en)
- Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages [[Paper]](https://arxiv.org/abs/2303.01037) [[Blog]](https://ai.googleblog.com/2023/03/universal-speech-model-usm-state-of-art.html)

## Large Multi-modal Models

### Healthcare Domain

- Bootstrapping Large Language Models for Radiology Report Generation [[Paper]] [[Code]](https://github.com/synlp/R2-LLM)
- Dietary Assessment with Multimodal ChatGPT: A Systematic Analysis [[Paper]](https://arxiv.org/abs/2312.08592)
- PLIP: A visual–language foundation model for pathology image analysis using medical Twitter [[Paper]](https://www.nature.com/articles/s41591-023-02504-3)
- LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day [[Paper]](https://arxiv.org/pdf/2306.00890.pdf)
- GPT-4 Technical Report [[Paper]](https://arxiv.org/pdf/2303.08774.pdf)
- Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning [[Paper]](https://www.nature.com/articles/s41551-022-00936-9)
- Contrastive Learning of Medical Visual Representations from Paired Images and Text [[Paper]](https://arxiv.org/pdf/2010.00747.pdf) [[Code]](https://github.com/edreisMD/ConVIRT-pytorch)
- Gloria: A multimodal global-local representation learning framework for labelefficient medical image recognition [[Paper]](https://ieeexplore.ieee.org/document/9710099) [[Code]](https://github.com/marshuang80/gloria)
- RAMM: Retrieval-augmented Biomedical Visual Question Answering with Multi-modal Pre-training [[Paper]](https://arxiv.org/abs/2303.00534)
- PubMedCLIP: How Much Does CLIP Benefit Visual Question Answering in the Medical Domain? [[Paper]](https://aclanthology.org/2023.findings-eacl.88/)
- SurgVLP: Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures[[Paper]](https://arxiv.org/abs/2307.15220) [[Code]](https://github.com/CAMMA-public/SurgVLP)


### General Domain

**Multi-modal Chatbot**
- The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision) [[Paper]](https://browse.arxiv.org/pdf/2309.17421.pdf)

**Representation learning**:

- Learning Transferable Visual Models From Natural Language Supervision [[paper]](https://proceedings.mlr.press/v139/radford21a.html)
- Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision [[paper]](https://proceedings.mlr.press/v139/jia21b.html)
- Florence: A New Foundation Model for Computer Vision [[paper]](http://arxiv.org/abs/2111.11432)
- Grounded Language-Image Pre-Training [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Grounded_Language-Image_Pre-Training_CVPR_2022_paper.html)
- WenLan: Bridging Vision and Language by Large-Scale Multi-Modal Pre-Training [[paper]](http://arxiv.org/abs/2103.06561)
- FLAVA: A Foundational Language and Vision Alignment Model [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Singh_FLAVA_A_Foundational_Language_and_Vision_Alignment_Model_CVPR_2022_paper.html)
- SimVLM: Simple Visual Language Model Pretraining with Weak Supervision [[paper]](https://openreview.net/forum?id=GUrhfTuf_3)
- FILIP: Fine-grained Interactive Language-Image Pre-Training [[paper]](https://openreview.net/forum?id=cpDhcsEDC2)
- Combined Scaling for Open-Vocabulary Image Classification [[paper]](http://arxiv.org/abs/2111.10050)
- BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation [[paper]](https://proceedings.mlr.press/v162/li22n.html)
- PaLI: A Jointly-Scaled Multilingual Language-Image Model [[paper]](http://arxiv.org/abs/2209.06794)
- Towards All-in-one Pre-training via Maximizing Multi-modal Mutual Information [[paper]](http://arxiv.org/abs/2211.09807)
- BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models [[paper]](http://arxiv.org/abs/2301.12597)
- Supervision Exists Everywhere: A Data Efficient Contrastive Language-Image Pre-training Paradigm [[paper]](https://openreview.net/forum?id=zq1iJkNk3uN)
- Language Is Not All You Need: Aligning Perception with Language Models [[paper]](http://arxiv.org/abs/2302.14045)
- PaLM-E: An Embodied Multimodal Language Model [[paper]](http://arxiv.org/abs/2303.03378)
- Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models [[paper]](http://arxiv.org/abs/2303.04671)

**Text-to-image generation**:

- Zero-Shot Text-to-Image Generation [[paper]](https://proceedings.mlr.press/v139/ramesh21a.html)
- High-Resolution Image Synthesis With Latent Diffusion Models [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)
- Hierarchical Text-Conditional Image Generation with CLIP Latents [[paper]](http://arxiv.org/abs/2204.06125)
- GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models [[paper]](https://proceedings.mlr.press/v162/nichol22a.html)
- Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding [[paper]](https://openreview.net/forum?id=08Yk-n5l2Al)
- Scaling Autoregressive Models for Content-Rich Text-to-Image Generation [[paper]](https://openreview.net/forum?id=AFDcYJKhND)

## Applications of Large AI Models in Healthcare

Note that some of the following models were not targeted at healthcare applications initially but may have the potential to be transferred to the healthcare domain or inspire future development.

### Bioinformatics

- GeneGPT: Augmenting Large Language Models with Domain Tools for Improved Access to Biomedical Information [[Paper]](https://arxiv.org/abs/2304.09667)
- Highly accurate protein structure prediction with AlphaFold [[Paper]](https://www.nature.com/articles/s41586-021-03819-2) [[Code]](https://github.com/deepmind/alphafold)
- Accurate prediction of protein structures and interactions using a three-track neural network [[Paper]](https://www.science.org/doi/full/10.1126/science.abj8754?casa_token=tleEHPOOSr8AAAAA%3AT0eToIMPW0oN1jjIGLs8aPyQK8qbcFIByjT1x4k90tvBAj03SZUzpEinCPe_t-g4ECmjJ9wlj8OwQBs)
- Protein complex prediction with AlphaFold-Multimer [[Paper]](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.abstract)
- FastFold: Reducing AlphaFold Training Time from 11 Days to 67 Hours [[Paper]](https://arxiv.org/abs/2203.00854) [[Code]](https://github.com/hpcaitech/fastfold)
- HelixFold: An Efficient Implementation of AlphaFold2 using PaddlePaddle [[Paper]](https://arxiv.org/abs/2207.05477) [[Code]](https://github.com/PaddlePaddle/PaddleHelix)
- Uni-Fold: An Open-Source Platform for Developing Protein Folding Models beyond AlphaFold [[Paper]](https://www.biorxiv.org/content/10.1101/2022.08.04.502811v3.abstract) [[Code]](https://github.com/dptech-corp/Uni-Fold)
- OpenFold: Retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization [[Paper]](https://www.biorxiv.org/content/10.1101/2022.11.20.517210v2.abstract) [[Code]](https://github.com/aqlaboratory/openfold)
- ManyFold: an efficient and flexible library for training and validating protein folding models [[Paper]](https://academic.oup.com/bioinformatics/article/39/1/btac773/6887136) [[Code]](https://github.com/instadeepai/manyfold)
- ColabFold: making protein folding accessible to all [[Paper]](https://www.nature.com/articles/s41592-022-01488-1) [[Code]](https://github.com/sokrypton/ColabFold)
- Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences [[Paper]](https://www.pnas.org/doi/abs/10.1073/pnas.2016239118) [[Code]](https://github.com/facebookresearch/esm)
- ProGen: Language Modeling for Protein Generation [[Paper]](https://arxiv.org/abs/2004.03497) [[Code]](https://github.com/lucidrains/progen)
- ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing [[Paper]](https://arxiv.org/abs/2007.06225) [[Code]](https://github.com/agemagician/ProtTrans)
- Evolutionary-scale prediction of atomic level protein structure with a language model [[Paper]](https://www.science.org/doi/full/10.1126/science.ade2574)
- High-resolution de novo structure prediction from primary sequence [[Paper]](https://www.biorxiv.org/content/10.1101/2022.07.21.500999v1.abstract) [[Code]](https://github.com/HeliXonProtein/OmegaFold)
- Single-sequence protein structure prediction using a language model and deep learning [[Paper]](https://www.nature.com/articles/s41587-022-01432-w)
- Improved the Protein Complex Prediction with Protein Language Models [[Paper]](https://www.biorxiv.org/content/10.1101/2022.09.15.508065v2.abstract)
- MSA Transformer [[Paper]](http://proceedings.mlr.press/v139/rao21a.html) [[Code]](https://github.com/The-AI-Summer/self-attention-cv)
- Deciphering antibody affinity maturation with language models and weakly supervised learning [[Paper]](https://arxiv.org/abs/2112.07782)
- xTrimoABFold: De novo Antibody Structure Prediction without MSA [[Paper]](https://arxiv.org/abs/2212.00735)
- scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data [[Paper]](https://arxiv.org/abs/2212.00735) [[Code]](https://github.com/TencentAILabHealthcare/scBERT)
- Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions [[Paper]](https://www.biorxiv.org/content/10.1101/2022.08.06.503062v2.abstract) [[Code]](https://github.com/ml4bio/rna-fm)
- E2Efold-3D: End-to-End Deep Learning Method for accurate de novo RNA 3D Structure Prediction [[Paper]](https://www.biorxiv.org/content/10.1101/2022.08.06.503062v2.abstract) [[Code]](https://github.com/ml4bio/rna-fm)
- SMILES-BERT: large scale unsupervised pre-training for molecular property prediction [[Paper]](https://par.nsf.gov/servlets/purl/10168888) [[Code]](https://github.com/uta-smile/SMILES-BERT)
- SMILES Transformer: Pre-trained molecular fingerprint for low data drug discovery [[Paper]](https://arxiv.org/abs/1911.04738) [[Code]](https://github.com/DSPsleeporg/smiles-transformer)
- MolBert: Molecular representation learning with language models and domain-relevant auxiliary tasks [[Paper]](https://arxiv.org/abs/2011.13230) [[Code]](https://github.com/BenevolentAI/MolBERT)
- AGBT: Algebraic graph-assisted bidirectional transformers for molecular property prediction [[Paper]](https://www.nature.com/articles/s41467-021-23720-w) [[Code]](https://github.com/ChenDdon/AGBTcode)
- GROVER: Self-supervised graph transformer on large-scale molecular data [[Paper]](https://arxiv.org/abs/2007.02835) [[Code]](https://github.com/tencent-ailab/grover)
- Molgpt: molecular generation using a transformer-decoder model [[Paper]](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00600) [[Code]](https://github.com/devalab/molgpt)
- A Model to Search for Synthesizable Molecules [[Paper]](https://arxiv.org/abs/1906.05221) [[Code]](https://github.com/john-bradshaw/molecule-chef)
- Transformer neural network for protein-specific de novo drug generation as a machine translation problem [[Paper]](https://www.nature.com/articles/s41598-020-79682-4)
- Deepconv-dti: Prediction of drug-target interactions via deep learning with convolution on protein sequences [[Paper]](https://arxiv.org/abs/1811.02114) [[Code]](https://github.com/GIST-CSBL/DeepConv-DTI)
- Graphdta: predicting drug–target binding affinity with graph neural networks [[Paper]](https://pubmed.ncbi.nlm.nih.gov/33119053/) [[Code]](https://github.com/thinng/GraphDTA)
- Moltrans: molecular interaction transformer for drug–target interaction prediction [[Paper]](https://arxiv.org/abs/2004.11424) [[Code]](https://github.com/kexinhuang12345/moltrans)
- Extracting Predictive Representations from Hundreds of Millions of Molecules [[Paper]](https://pubs.acs.org/doi/10.1021/acs.jpclett.1c03058) [[Code]](https://github.com/WeilabMSU/PretrainModels)
- ADMETlab 2.0: an integrated online platform for accurate and comprehensive predictions of ADMET properties [[Project]](https://admetmesh.scbdd.com/) [[Paper]](https://pubmed.ncbi.nlm.nih.gov/33893803/)
- MPG: Learn molecular representations from large-scale unlabeled molecules for drug discovery [[Paper]](https://arxiv.org/abs/2012.11175)
- MG-BERT: leveraging unsupervised atomic representation learning for molecular property prediction [[Paper]](https://academic.oup.com/bib/article-abstract/22/6/bbab152/6265201?redirectedFrom=fulltext) [[Code]](https://github.com/ParishadBehnam/MG-BERT)
- PanGu Drug Model: Learn a Molecule Like a Human [[Project]](http://www.pangu-drug.com/) [[Paper]](https://www.biorxiv.org/content/10.1101/2022.03.31.485886v1.full)
- DrugBAN: Interpretable bilinear attention network with domain adaptation improves drug–target prediction [[Paper]](https://www.nature.com/articles/s42256-022-00605-1) [[Code]](https://github.com/peizhenbai/DrugBAN)
- DrugOOD: Out-of-Distribution (OOD) Dataset Curator and Benchmark for AI-aided Drug Discovery [[Paper]](https://arxiv.org/abs/2201.09637) [[Code]](https://github.com/tencent-ailab/DrugOOD)

### Medical Diagnosis

- VisionFM: A Multi-Modal Multi-Task Vision Foundation Model for Generalist Ophthalmic Artificial Intelligence [[Paper]](https://arxiv.org/abs/2310.04992)
- RETFound: A foundation model for generalizable disease detection from retinal images [[Paper]](https://www.nature.com/articles/s41586-023-06555-x)
- LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day [[Paper]](https://arxiv.org/pdf/2306.00890.pdf)
- Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning [[Paper]](https://www.nature.com/articles/s41551-022-00936-9)
- ChatCAD: Interactive Computer-Aided Diagnosis on Medical Image using Large Language Models [[Paper]](https://arxiv.org/pdf/2302.07257.pdf) [[Code]](https://github.com/zhaozh10/ChatCAD)
- BEHRT: Transformer for Electronic Health Records [[Paper]](https://www.nature.com/articles/s41598-020-62922-y)
- Federated Learning of Medical Concepts Embedding using BEHRT [[Paper]](https://arxiv.org/abs/2305.13052) [[Code]](https://github.com/nadavlab/FederatedBEHRT)
- Med-BERT: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction [[Paper]](https://www.nature.com/articles/s41746-021-00455-y)
- CPLLM: Clinical Prediction with Large Language Models [[Paper]](https://arxiv.org/abs/2309.11295) [[Code]](https://github.com/nadavlab/CPLLM)
- RadBERT: Adapting Transformer-based Language Models to Radiology [[paper]](https://pubs.rsna.org/doi/epdf/10.1148/ryai.210258) [[HuggingFace]](https://huggingface.co/UCSD-VA-health/RadBERT-RoBERTa-4m)
- ChatCAD+: Towards a Universal and Reliable Interactive CAD using LLMs [[paper]](https://arxiv.org/abs/2305.15964) [[Code]](https://github.com/zhaozh10/ChatCAD)

### Medical Imaging

- VisionFM: A Multi-Modal Multi-Task Vision Foundation Model for Generalist Ophthalmic Artificial Intelligence [[Paper]](https://arxiv.org/abs/2310.04992)
- RETFound: A foundation model for generalizable disease detection from retinal images [[Paper]](https://www.nature.com/articles/s41586-023-06555-x)
- Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning [[Paper]](https://www.nature.com/articles/s41551-022-00936-9)
- Med3d: Transfer learning for 3d medical image analysis [[Paper]](https://arxiv.org/abs/1904.00625) [[Code]](https://github.com/Tencent/MedicalNet)
- Models genesis: Generic autodidactic models for 3d medical image analysis [[Paper]](https://arxiv.org/abs/1908.06912) [[Code]](https://github.com/MrGiovanni/ModelsGenesis)
- MICLe: Big self-supervised models advance medical image classifications [[Paper]](https://arxiv.org/abs/2101.05224) [[Code]](https://github.com/rjrobben/MICLe_pytorch)
- C2l: Comparing to Learn: Surpassing ImageNet Pretraining on Radiographs By Comparing Image Representations [[Paper]](https://arxiv.org/abs/2007.07423) [[Code]](https://github.com/funnyzhou/C2L_MICCAI2020)
- ConVIRT: Contrastive learning of medical visual representations from paired images and text [[Paper]](https://arxiv.org/pdf/2303.11032.pdf) [[Code]](https://github.com/edreisMD/ConVIRT-pytorch)
- Gloria: A multimodal global-local representation learning framework for labelefficient medical image recognition [[Paper]](https://ieeexplore.ieee.org/document/9710099) [[Code]](https://github.com/marshuang80/gloria)
- MoCo-CXR: MoCo Pretraining Improves Representation and Transferability of Chest X-ray Models [[Paper]](https://arxiv.org/abs/2010.05352) [[Code]](https://github.com/stanfordmlgroup/MoCo-CXR)
- Transunet: Transformers make strong encoders for medical image segmentation [[Paper]](https://arxiv.org/abs/2102.04306) [[Code]](https://github.com/Beckschen/TransUNet)
- Transfuse: Fusing transformers and cnns for medical image segmentation [[Paper]](https://arxiv.org/abs/2102.08005) [[Code]](https://github.com/Rayicer/TransFuse)
- Medical transformer: Gated axial-attention for medical image segmentation [[Paper]](https://arxiv.org/abs/2102.10662) [[Code]](https://github.com/jeya-maria-jose/Medical-Transformer)
- UNETR: Transformers for 3D Medical Image Segmentation [[Paper]](https://arxiv.org/abs/2103.10504) [[Code]](https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV)
- Cotr: Efficiently bridging cnn and transformer for 3d medical image segmentation [[Paper]](https://arxiv.org/abs/2103.03024) [[Code]](https://github.com/YtongXie/CoTr)
- Swin-unet: Unet-like pure transformer for medical image segmentation [[Paper]](https://arxiv.org/abs/2105.05537) [[Code]](https://github.com/HuCaoFighting/Swin-Unet)
- SAM4Med: Generalist Vision Foundation Models for Medical Imaging: A Case Study of Segment Anything Model on Zero-Shot Medical Segmentation [[Paper]](https://arxiv.org/pdf/2304.12637.pdf)

### Medical Informatics

- Med-PaLM 2: Towards Expert-Level Medical Question Answering with Large Language Models [[Paper]](https://arxiv.org/pdf/2305.09617.pdf)
- DeID-GPT: Zero-shot Medical Text De-Identification by GPT-4 [[Paper]](https://arxiv.org/pdf/2303.11032.pdf)
- Capabilities of GPT-4 on Medical Challenge Problems [[Paper]](https://arxiv.org/pdf/2303.13375.pdf)
- BioBERT: a pre-trained biomedical language representation model for biomedical text mining [[Paper]](https://arxiv.org/pdf/1901.08746.pdf)
- Publicly Available Clinical BERT Embeddings [[Paper]](https://arxiv.org/pdf/1904.03323.pdf)
- BioMegatron: Larger Biomedical Domain Language Model [[Paper]](https://arxiv.org/pdf/2010.06060.pdf)
- Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks [[Paper]](https://aclanthology.org/2020.acl-main.740.pdf)
- Med-BERT: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction [[Paper]](https://www.nature.com/articles/s41746-021-00455-y)
- CPLLM: Clinical Prediction with Large Language Models [[Paper]](https://arxiv.org/abs/2309.11295) [[Code]](https://github.com/nadavlab/CPLLM)
- BioELECTRA:Pretrained Biomedical text Encoder using Discriminators [[Paper]](https://aclanthology.org/2021.bionlp-1.16.pdf)
- LinkBERT: Pretraining Language Models with Document Links [[Paper]](https://arxiv.org/pdf/2203.15827.pdf)
- BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining [[Paper]](https://arxiv.org/pdf/2210.10341.pdf)
- Large Language Models Encode Clinical Knowledge [[Paper]](https://arxiv.org/pdf/2212.13138.pdf)
- A large language model for electronic health records [[Paper]](https://www.nature.com/articles/s41746-022-00742-2)
- Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing [[Paper]](https://arxiv.org/pdf/2007.15779.pdf)
- BEHRT: Transformer for Electronic Health Records [[Paper]](https://www.nature.com/articles/s41598-020-62922-y)
- Federated Learning of Medical Concepts Embedding using BEHRT [[Paper]](https://arxiv.org/abs/2305.13052) [[Code]](https://github.com/nadavlab/FederatedBEHRT)

### Medical Education

- GPT-4 Technical Report [[Paper]](https://arxiv.org/pdf/2303.08774.pdf)
- Empowering Beginners in Bioinformatics with ChatGPT [[Paper]](https://www.biorxiv.org/content/10.1101/2023.03.07.531414v1)

### Public Health

- Dietary Assessment with Multimodal ChatGPT: A Systematic Analysis [[Paper]](https://arxiv.org/abs/2312.08592)
- Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning [[Paper]](https://www.nature.com/articles/s41551-022-00936-9)
- Clustering Egocentric Images in Passive Dietary Monitoring with Self-Supervised Learning [[Paper]](https://arxiv.org/pdf/2208.12160.pdf)
- ClimaX: A foundation model for weather and climate [[Paper]](https://arxiv.org/pdf/2301.10343.pdf)

### Medical Robotics

- EndoFM: Foundation Model for Endoscopy Video Analysis via Large-scale Self-supervised Pre-train [[Paper]](https://arxiv.org/abs/2306.16741) [[Code]](https://github.com/med-air/Endo-FM)
- Decision Transformer: Reinforcement Learning via Sequence Modeling [[Paper]](https://arxiv.org/abs/2106.01345) [[Code]](https://github.com/kzl/decision-transformer)
- R3M: A Universal Visual Representation for Robot Manipulation [[Paper]](https://arxiv.org/abs/2203.12601) [[Project]](https://sites.google.com/view/robot-r3m/) [[Code]](https://github.com/facebookresearch/r3m)
- MimicPlay: Long-Horizon Imitation Learning by Watching Human Play [[Paper]](https://arxiv.org/abs/2302.12422) [[Project]](https://mimic-play.github.io/)
- PaLM-E: An Embodied Multimodal Language Model [[Paper]](https://arxiv.org/abs/2303.03378) [[Project]](https://palm-e.github.io/) [[Blog]](https://ai.googleblog.com/2023/03/palm-e-embodied-multimodal-language.html)
- A Generalist Agent [[Paper]](https://arxiv.org/abs/2205.06175) [[Blog]](https://www.deepmind.com/blog/a-generalist-agent)
- CLIPort: What and Where Pathways for Robotic Manipulation [[Paper]](https://arxiv.org/abs/2109.12098) [[Project]](https://cliport.github.io/) [[Code]](https://github.com/cliport/cliport)
- Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation [[Paper]](https://arxiv.org/abs/2209.05451) [[Project]](https://peract.github.io/) [[Code]](https://github.com/peract/peract)
- Do As I Can, Not As I Say: Grounding Language in Robotic Affordances [[Paper]](https://arxiv.org/abs/2204.01691) [[Project]](https://say-can.github.io/) [[Code]](https://github.com/google-research/google-research/tree/master/saycan)
- VIMA: General Robot Manipulation with Multimodal Prompts [[Paper]](https://arxiv.org/abs/2210.03094) [[Project]](https://vimalabs.github.io/) [[Code]](https://github.com/vimalabs/VIMA)
- RT-1: Robotics Transformer for Real-World Control at Scale [[Paper]](https://arxiv.org/abs/2212.06817) [[Project]](https://robotics-transformer.github.io/) [[Code]](https://github.com/google-research/robotics_transformer)
- ChatGPT for Robotics: Design Principles and Model Abilities [[Paper]](https://www.microsoft.com/en-us/research/uploads/prod/2023/02/ChatGPT___Robotics.pdf) [[Blog]](https://www.microsoft.com/en-us/research/group/autonomous-systems-group-robotics/articles/chatgpt-for-robotics/) [[Code]](https://github.com/microsoft/PromptCraft-Robotics)


## AI Legislation

- AI Act (EU) [[Source]](https://artificialintelligenceact.eu/)
- A pro-innovation approach to AI regulation (UK) [[Source]](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1146542/a_pro-innovation_approach_to_AI_regulation.pdf)
- Blueprint for an AI Bill of Rights (USA) [[Source]](https://www.whitehouse.gov/ostp/ai-bill-of-rights/)
- AI Risk Management Framework (USA) [[Source]](https://www.nist.gov/itl/ai-risk-management-framework)
- Provisions on the Administration of Deep Synthesis Internet Information Services (China) [[Source]](https://www.chinalawtranslate.com/en/deep-synthesis/)
- Interim Measures for the Management of Generative Artificial Intelligence Services (China) [[Source]](https://www.chinalawtranslate.com/en/generative-ai-interim/)

## Large-scale Datasets in Biomedical and Health Informatics
### Open Source

| Dataset                                                      | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Big Fantastic Datasbase](https://bfd.mmseqs.com/)           | 2.1 B protein sequences, 393 B amino acids                   |
| [Observed Antibody Space](https://opig.stats.ox.ac.uk/webapps/oas/) | 558 M antibody sequences                                     |
| [RNAcentral](https://rnacentral.org/)                        | 34 M ncRNA sequences, 22 M secondary structure               |
| [ZINC20](https://zinc20.docking.org/)                        | 1.4B compounds from 310 catalogs from 150 companies          |
| [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/)  | 65K patients, 337K chest X-ray images and 227K radiology reports |
| [MedMNIST v2](https://medmnist.com/)                         | 708K 2D medical images, 10K 3D medical images                |
| [Medical Meadow](https://github.com/kbressem/medAlpaca)      | 1.5M data points containing a wide range of medical language processing tasks |
| [Endo-FM database](https://github.com/med-air/Endo-FM)       | 33K endoscopic videos, up to 5M frames                       |
| [SurgVLP database](https://github.com/CAMMA-public/SurgVLP)       | 25K laparoscopic video-text pairs from 1k surgical lecture videos |

### Private or Upon Approval
| Dataset                                                      | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Mount Sinai ECG Data](https://www.nature.com/articles/s41746-023-00840-9)           | 2.1 M patients, containing 8.5 M discrete ECG recordings|
| [Google DR Dev. Dataset](https://jamanetwork.com/journals/jama/fullarticle/2588763) | 239 K unique individuals, 1.6 M fundus images |
| [UF Health IDR Clinical Note Database](https://www.nature.com/articles/s41746-022-00742-2) | 290 M clinical notes, with up to 82 B medical words |
| [Clinical Practice Research Datalink](https://academic.oup.com/ije/article/44/3/827/632531) | 11.3 M patients covering data on demographics, symptoms, diagnoses, etc |


