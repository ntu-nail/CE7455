# CE7455: Deep Learning For Natural Language Processing

# Course Objectives

Natural language processing (NLP) is one of the most important fields in artificial intelligence (AI). It has become very crucial in the information age because most of the information is in the form of unstructured text. NLP technologies are applied everywhere as people communicate mostly in language: language translation, web search, customer support, emails, forums, advertisement, radiology reports, to name a few.

There are several core NLP tasks and machine learning models behind NLP applications. Deep learning, a sub-field of machine learning, has recently brought a paradigm shift from traditional task-specific feature engineering to end-to-end systems and has obtained high performance across many different NLP tasks and downstream applications. Tech companies like Google, Baidu, Alibaba, Apple, Amazon, Facebook, Tencent, and Microsoft are now actively working on deep learning methods to improve their products. For example, Google recently replaced its traditional statistical machine translation and speech-recognition systems with systems based on deep learning methods.

**Optional Textbooks**

- Deep Learning by Goodfellow, Bengio, and Courville [free online](http://www.deeplearningbook.org/)
- Machine Learning — A Probabilistic Perspective by Kevin Murphy [online](https://doc.lagout.org/science/Artificial%20Intelligence/Machine%20learning/Machine%20Learning_%20A%20Probabilistic%20Perspective%20%5BMurphy%202012-08-24%5D.pdf)
- Natural Language Processing by Jacob Eisenstein [free online](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf)
- Speech and Language Processing by Dan Jurafsky and James H. Martin [(3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/)

# Intended Learning Outcomes

In this course, students will learn state-of-the-art deep learning methods for NLP. Through lectures and practical assignments, students will learn the necessary tricks for making their models work on practical problems. They will learn to implement, and possibly invent their deep learning models using available deep learning libraries like [Pytorch](https://pytorch.org/).

**Our Approach**

- Thorough and Detailed: How to write from scratch, debug, and train deep neural models

- State of the art: Most lecture materials are new from the research world in the past 1-5 years.

- Practical: Focus on practical techniques for training the models on GPUs.

- Fun: Cover exciting new advancements in NLP (e.g., Transformer, BERT).

# Assessment Approach

**Weekly Workload**

- Lecture and practical problems implemented in PyTorch.
- There will be NO office hours.

**Assignments (individually graded)**

- Two (2) assignments will contribute to ***2 * 25% = 50%*** of the total assessment.
- Students will be graded individually on the assignments. They will be allowed to discuss with each other on the homework assignments, but they are required to submit individual write-ups and coding exercises.

**Final Project (Group work but individually graded)**

- There will be a final project contributing to the remaining 50% of the total coursework assessment.
  - ***3–5*** students per group
  - Project proposal: ***5%***, presentation: ***15%***, report: ***30%***
- The project will be a group work. Students will be graded individually, depending on their contribution to the group. The final project presentation will ensure the student’s understanding of the project.

# Course Prerequisites

- Proficiency in Python (using Numpy and PyTorch). There is a lecture for those who are not familiar with Python.
- Linear Algebra, basic Probability and Statistics
- Machine Learning basics

# Teaching

## Instructors

<p align="center" width="100%"><a href="https://personal.ntu.edu.sg/wangwy/">Wang Wenya (Part 1)</a></p>

<p align="center" width="100%"><a href="https://tuanluu.github.io/">Luu Anh Tuan (Part 2)</a></p>

## Teaching Assistants

<p align="center" width="100%">He Qiyuan (Part 1)</p>
<p align="center" width="100%">qiyuan001@e.ntu.edu.sg</p>

<p align="center" width="100%">Nguyen Tran Cong Duy (Part 2)</p>
<p align="center" width="100%">nguyentr003@e.ntu.edu.sg</p>

# Schedule & Course Content

## Week 1: Introduction

[Lecture Slide](https://drive.google.com/file/d/1cVAxrntk1Q4R_LC1zFQufkfyakG21E-y/view?usp=sharing)

### Lecture Content

- What is Natural Language Processing?
- Why is language understanding difficult?
- What is Deep Learning?
- Deep learning vs. other machine learning methods?
- Why deep learning for NLP?
- Applications of deep learning to NLP
- Knowing the target group (background, field of study, programming experience)
- Expectation from the course

### Python & PyTorch Basics

- Programming in Python

  - Jupiter Notebook and [Google Colab](https://colab.research.google.com/drive/16pBJQePbqkz3QFV54L4NIkOn1kwpuRrj)
  - [Introduction to Python](https://colab.research.google.com/drive/1bQG32CFoMZ-jBk02uaFon60tER3yFx4c)
  - Deep Learning Frameworks
  - Why Pytorch?
  - [Deep learning with PyTorch](https://drive.google.com/file/d/1c33y8bkdr7SJ_I8-wmqTAhld-y7KcspA/view?usp=sharing)
- [Supplementary]
  - Numerical programming with Numpy/Scipy - [Numpy intro](https://drive.google.com/file/d/1cUzRzQGURrCKes8XynvTTA4Zvl_gUJdc/view?usp=sharing)
  - Numerical programming with Pytorch - [Pytorch intro](https://drive.google.com/file/d/18cgPOj2QKQN0WR9_vXoz6BoravvS9mTm/view?usp=sharing)


## Week 2: Machine Learning Basics

[Lecture Slide](https://docs.google.com/presentation/d/1oRnv9yGMOm2vbznzS7ii4TB7PCxVD1bawtAGBAj0kwU/edit?usp=sharing)

### Lecture Content

- What is Machine Learning?
- Machine learning for text classification
- Naive Bayes
- Logistic Regression
- Multi-class classification
- Gradient-based optimization

### Practical exercise with Pytorch

- [Deep learning with PyTorch](https://colab.research.google.com/drive/1aZVfsPUko-ugt1TVCmRwqGJXlxEJVaTq?usp=sharing)
- [Linear Regression](https://colab.research.google.com/drive/12QpBf7x_Jt6-zypN4OrUFFHXz1u6CmYe?usp=sharing)
- [Logistic Regression](https://colab.research.google.com/drive/1nTrYW5dUu6WO9cx7SGEvP9oX7qRbsGJk?usp=sharing)
- [Supplementary]
  - Numerical programming with Pytorch - [Pytorch intro](https://drive.google.com/file/d/18cgPOj2QKQN0WR9_vXoz6BoravvS9mTm/view?usp=sharing)



## Week 3: Neural Networks & Optimization Basics

[Lecture Slide](https://drive.google.com/file/d/1jc9T1-6WJ6GizPyRXKq7boLl0G9DgH9D/view?usp=sharing)

### Lecture Content

- From Logistic Regression to Feed-forward NN
  - Activation functions
- SGD with Backpropagation
- Adaptive SGD (adagrad, adam, RMSProp)
- Regularization (Dropout, Batch normalization, L1/L2 norm, Gradient clipping)

### Practical exercise with Pytorch

- [Deep learning with PyTorch](https://colab.research.google.com/drive/1aZVfsPUko-ugt1TVCmRwqGJXlxEJVaTq?usp=sharing)
- [Numpy notebook](https://colab.research.google.com/drive/1IAonxZnZjJb0_xUVWHt5atIxaI5GTJQ2) [Pytorch notebook](https://colab.research.google.com/drive/1YzZrMAmJ3hjvJfNIdGxae9kxGABG6yaT)
  - Backpropagation
  - Dropout
  - Batch normalization
  - Initialization
  - Gradient clipping


## Week 4: Word Vectors 

[Lecture Slide](https://drive.google.com/file/d/1uLkxAm1vDEPBvYn5lOkMHTtPxSiMOyU5/view?usp=sharing)


[Project Proposal Instruction](https://docs.google.com/document/d/1i6QlzNX-HDjkW5HwRTLX8lYC6F4tCwhO40oqUItMP_s/edit?usp=sharing)

[Group Allocation (form your own group)](https://docs.google.com/spreadsheets/d/13zTwv7U5AFvr7SfBUFFT2W7tWx8PCJm1kVUqpVPFqWg/edit?usp=sharing)


### Lecture Content

- Word meaning
- Distributed representation of words
- Word2Vec models (Skip-gram, CBOW)
- Negative sampling
- Glove
- FastText
- Evaluating word vectors
  - Intrinsic evaluation
  - Extrinsic evaluation
- Cross-lingual word embeddings

### Practical exercise with Pytorch

[Skip-gram training](https://colab.research.google.com/drive/164dB-Vemzwavf1ffqDDVNtx7Y5VtcmQh)

### Suggested Readings

- Word2Vec Tutorial - The Skip-Gram Model, [blog](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) - Original word2vec paper
- [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) - negative sampling paper
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
- [FastText: Enriching Word Vectors with Subword Information](https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00051?mobileUi=0)
- [Linguistic Regularities in Sparse and Explicit Word Representations.](https://levyomer.files.wordpress.com/2014/04/linguistic-regularities-in-sparse-and-explicit-word-representations-conll-2014.pdf)
- [Neural Word Embeddings as Implicit Matrix Factorization.](https://arxiv.org/abs/1702.02098)
- [Survey on Cross-lingual embedding methods](https://arxiv.org/abs/1706.04902)
- [Slides on Cross-lingual embedding](https://www.dropbox.com/s/3eq5apr75yrz9ix/Cross-lingual%20word%20embeddings%20and%20beyond.pdf?dl=0)
- [Adversarial autoencoder for unsupervised word translation](https://arxiv.org/abs/1904.04116)
- [Evaluating Cross-Lingual Word Embeddings](https://www.aclweb.org/anthology/P19-1070)


## Week 5: Window-based Approach and Convolutional Nets

[Lecture Slide](https://drive.google.com/file/d/1MpVGB3UBvPhUdcuj0r_ui_WkoMJ5IBUE/view?usp=sharing)

### Lecture Content

- Classification tasks in NLP
- Window-based Approach for language modeling
- Window-based Approach for NER, POS tagging, and Chunking
- Convolutional Neural Net for NLP
- Loss Functions

### Suggested Readings

- [Linear Algebraic Structure of Word Senses, with Applications to Polysemy](https://transacl.org/ojs/index.php/tacl/article/viewFile/1346/320)
- [Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://www.aclweb.org/anthology/Q15-1016/)
- [Natural Language Processing (Almost) from Scratch](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)
- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- [Fast and Accurate Entity Recognition with Iterated Dilated Convolutions](https://arxiv.org/abs/1702.02098)

### Practical exercise with Pytorch

[Named Entity Recognition](https://colab.research.google.com/drive/1aDFQtSCYBpgRwkj9eq8Rao9eq5_68C31)

[CNN for text classification](https://colab.research.google.com/drive/1r3qUTV0AQBV9cUAF0GML6Yu3aMfPdBGQ#scrollTo=mc6ZP_xosSil)


## Week 6: Recurrent Neural Nets

[Lecture Slide](https://drive.google.com/file/d/1jGUvuEp1rhNXsXK-BBpnlmmucqGt4uLc/view?usp=sharing)


Assignment 1 is out [here](https://docs.google.com/document/d/1oVHB2kby-G9ikIdS6dxnIMiDz1gPbQw4dd7_QghXt7M/edit?usp=sharing). **Deadline: 23 March 2025**.

### Lecture Content


- Language modeling with RNNs
- Backpropagation through time
- Text generation with RNN LM
- Sequence labeling with RNNs
- Sequence classification with RNNs
- Issues with Vanilla RNNs
- Gated Recurrent Units (GRUs) and LSTMs
- Bidirectional RNNs
- Multi-layer RNNs
- Recursive Neural Nets

### Practical exercise with Pytorch (RNN for POS Tagging)

- [POS Tagging](https://colab.research.google.com/drive/1a_0JGhg-N9pItzUDwfULtuxdbnh1m1y8)

### Suggested Readings

- [N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
- [Karpathy’s nice blog on Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Building an Efficient Neural Language Model](https://research.fb.com/building-an-efficient-neural-language-model-over-a-billion-words/)
- [On the difficulty of training recurrent neural networks](http://proceedings.mlr.press/v28/pascanu13.pdf)
- [Colah’s blog on LSTMs/GRUs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Neural Architectures for Named Entity Recognition](https://www.aclweb.org/anthology/N16-1030/)
- [Fine-grained Opinion Mining with Recurrent Neural Networks and Word Embeddings](https://www.aclweb.org/anthology/D15-1168/)



## Week 7: Machine translation and Seq2Seq Models

[Lecture Slide](https://drive.google.com/file/d/1BxNxakiFSN9FYEbA2VKSuHQSmWA1sXyz/view?usp=sharing)

### Lecture Content

- Machine translation
  - Early days (1950s)
  - Statistical machine translation or SMT (1990-2010)
  - Alignment in SMT
  - Neural machine translation or NMT (2014 - )
- Encoder-decoder model for NMT
- Advantages and disadvantages of NMT
- Decoding strategies such as beam search and sampling
- MT evaluation
- Other evaluation methods for language generation
- Review

### Suggested Readings

- [Statistical Machine Translation slides, CS224n 2015 (lectures 2/3/4)](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1162/syllabus.shtml)
- [Sequence to Sequence Learning with Neural Networks (original seq2seq NMT paper)](https://arxiv.org/pdf/1409.3215.pdf)
- [Statistical Machine Translation (book by Philipp Koehn)](https://www.cambridge.org/core/books/statistical-machine-translation/94EADF9F680558E13BE759997553CDE5)
- [A Neural Conversational Model](https://arxiv.org/abs/1506.05869)
- [BLEU (original paper)](https://www.aclweb.org/anthology/P02-1040.pdf)

### Practical exercise with Pytorch

- [Neural Machine Translation](https://colab.research.google.com/drive/1htCUnTIFY9NAhbANuuZEeHGxvfPGHBdK)


## Week 8: Seq2Seg Models, Attentions, Subwords

[Lecture Slide](https://drive.google.com/file/d/14stMogPaHp6-s2bQqo-eR1FvzJ5Dt_T1/view?usp=share_link)

### Lecture Content

- Information bottleneck issue with vanilla Seq2Seq
- Attention to the rescue
- Details of attention mechanism
- Sub-word models
- Byte-pair encoding
- Hybrid models


### Practical exercise with Pytorch

- [Neural machine translation tutorial in pytorch](https://colab.research.google.com/drive/1cYyBxmdjFjKls0CEsPc8WIHfxPDy4eSq)


### Suggested Readings

- [Neural Machine Translation by Jointly Learning to Align and Translate (original seq2seq+attention paper)](https://arxiv.org/pdf/1409.0473.pdf)
- [Effective Approaches to Attention-based Neural Machine Translation](https://nlp.stanford.edu/~lmthang/data/papers/emnlp15_attn.pdf)
- [Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models](https://arxiv.org/abs/1604.00788)



## Week 9: Seq2Seq Variants and Transformer

[Lecture Slide](https://drive.google.com/file/d/1wWxMg_tIVdShRccSyH-v38g-LKmOcO3L/view?usp=sharing)

### Lecture Content

- Seq2Seq Variants (Pointer nets, Pointer Generator Nets)
  - Machine Translation
  - Summarization
- Transformer architecture
  - Self-attention
  - Positional encoding
  - Multi-head attention

[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)

### Suggested Readings

- [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
- [Pointer Networks](https://papers.nips.cc/paper/5866-pointer-networks)
- [Stack-Pointer Networks for Dependency Parsing](https://www.aclweb.org/anthology/P18-1130.pdf)
- [A Unified Linear-Time Framework for Sentence-Level Discourse Parsing](https://arxiv.org/abs/1905.05682)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Resurrecting Submodularity in Neural Abstractive Summarization](https://arxiv.org/abs/1911.03014)



## Week 10: Contextual embeddings and self-supervised learning

[Lecture Slide](https://drive.google.com/file/d/18upNeGM8YdUMl9dn-FqfVrIeFVfb3mtb/view?usp=share_link)



### Lecture Content

- Why semi-supervsied?
- Semisupervised learning dimensions
- Pre-training and fine-tuning methods

  - CoVe
  - TagLM
  - ELMo
  - GPT
  - ULMfit
  - BERT
  - BART
- Evaluation benchmarks

  - GLUE
  - SQuAD
  - NER
  - SuperGLUE
  - XNLI

[Pre-train Fine-tune with HF](https://colab.research.google.com/drive/1L_hwnQISoIBrH7W_r83I62hJ4FBlfNsz?usp=sharing)

### Suggested Readings

- [Cove Paper](https://arxiv.org/pdf/1708.00107.pdf)
- [ULMFit paper](https://arxiv.org/abs/1801.06146)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [ELMo paper](https://arxiv.org/abs/1802.05365)
- [BART paper](https://arxiv.org/abs/1910.13461)


## Week 11: Large Language Models & Multilingual NLP


Assignment 2 is out [here](https://docs.google.com/document/d/1AonpEdaFc3Tgu0RdEux-ZMaJ0-RHOrPp/edit?usp=sharing&ouid=118265590333925180950&rtpof=true&sd=true). **Deadline: 23 Apr 2025, 11:59pm**.

[Final project report instruction](https://drive.google.com/file/d/1zV7cV4hOMWaq2NcvoHumkx6U16HDATeL/view?usp=sharing)

[Lecture Slide](https://drive.google.com/file/d/1L5H0_e7tVQw-lBELcejppVs2DTY6hb2b/view?usp=share_link)

### Lecture Content

- Large Language Models
- Examples of Large Language Models
- Multilingual NLP
  - Why do we need Multilingual NLP?
  - Low-resource NLP
  - Cross-lingual models
  - Multilingual models

### Suggested Readings

- [XLM paper](https://arxiv.org/abs/1901.07291)
- [Transformer XL paper](https://arxiv.org/pdf/1901.02860.pdf)
- [XLNet paper](https://arxiv.org/abs/1906.08237)
- [mBART paper](https://arxiv.org/abs/2001.08210)
<!--

## Week 12: Bias, Robustness, Hallucination, Multimodal NLP & Recap

[Lecture Slide](https://drive.google.com/file/d/1bH3rNigKxn3mhJ8KKUeaTi6mMLiRL0SH/view?usp=drive_link)

### Lecture Content

- Bias Problem in Deep Learning for NLP
- Robustness of NLP Deep Learning Models
- Hallucination of LLMs
- Multimodal NLP
- Recap

### Suggested Readings

- [Bias in NLP](https://aclanthology.org/2021.acl-long.416.pdf)
- [Robustness NLP](https://arxiv.org/pdf/2107.13541.pdf)
- [Hallucination LLMs](https://arxiv.org/pdf/2309.01219.pdf?trk=public_post_comment-text)
- [Multimodal NLP](https://aclanthology.org/2022.lrec-1.738.pdf)
-->
