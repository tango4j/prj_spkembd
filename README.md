# spkembd
Speaker embedding extractor for various tasks.
Open-source speaker embedding extractor.

## Reference Publications

### 1. Speaker Verification Papers
[2015, George et al., Google, End-to-End Text-Dependent Speaker Verification](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/44681.pdf)

**Application**: Text Dependent  
**Feature**: 40 dim Filterbank   
**Neural Net Architecture**: Maxout-DNN(4-layer), embedding layer before softmax  
**Loss Function**: Cross entropy Loss  
**Normalization**: L2 Norm 
**Dataset Size**: 646 speakers  
**Baseline**: i-vector

[2018, Li et al., Google, GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION](https://arxiv.org/pdf/1710.10467.pdf)


**Application**: Text Independent and Independent("OK Google")  
**Feature**: 40 dim Filterbank   
**Neural Net Architecture**: 3 layer LSTM   
**Loss Function**: GE2E (Generalized end-to-end loss)   
**Normalization**: L2 Norm  
**Window Size**: 1.6 second overlap 50%, element wise average  
**Dataset Size**: 1000 speakers, 6.3 enrollment utterances, 7.2 evaluation utterances  
**Baseline**: TE2E (Tuple based end-to-end loss)  

[2017, Chao et al., Baidu, End-to-End Neural Speaker Embedding System](https://arxiv.org/pdf/1705.02304.pdf)

**Application**: Text Dependent and Independent  
**Feature**: 64 dim Fbank  
**Neural Net Architecture**: ResNet CNN and GRU  
**Pre-training**: Yes (softmax pre-training)  
**Loss Function**: Triplet loss with cosine distance metric  
**Feature Normalization**: Zero mean unit variance  
**Dataset**: Mandarin and English (not public)  
**Dataset Size**: 250,000 speakers  
**Baseline**: DNN i-vector system  
**GPUs**: 16 K40 GPUs  

[2017, David et al, JHU, Deep Neural Network Embeddings for Text-Independent Speaker Verification](https://www.isca-speech.org/archive/Interspeech_2017/pdfs/0620.PDF)


### 2. Loss Function Papers

[2015, Florian et al., Google, (Triplet loss paper)FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)

## Dependency

[kaldi_io](https://github.com/vesis84/kaldi-io-for-python.git)
numpy\
Pytorch\
awef

## Dataset 


## Models
### Feature

## Performance Evaluation
