# rnn-sequence-generation

This project aims to replicate the results of the paper; 
A. Graves, 2013, Generating sequences with recurrent neural networks, CoRR journal, http://arxiv.org/abs/1308.0850. 

This project is written in Julia language using Knet. 

The project focuses on the usage of Long Short-term Memory recurrent neural networks in sequence generation, where
the testing is done with text and online handwriting(data collected from a smart board). Idea behind RNN’s 
(including LSTM) is to feed the output of one network cell to another one. The strength of this LSTM model is it’s 
long memory and robustness to high dimensions. 