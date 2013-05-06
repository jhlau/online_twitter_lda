On-line LDA
===========

Original author: Shuyo NAKATANI

Modified by: Jey Han LAU

License: MIT license: http://opensource.org/licenses/MIT

Details of this code are described in the following paper:

  Lau, Jey Han, Nigel Collier and Timothy Baldwin (2012) On-line Trend
  Analysis with Topic Models: #twitter trends detection topic model online, In
  Proceedings of the 24th International Conference on Computational
  Linguistics (COLING 2012), Mumbai, India.

Please cite this paper when using the code.


Files in the Directory
======================
input: sample input files for testing the program
lda.py: main module that does the topic modelling
README.txt: this readme file
run_lda.sh: script to run the online lda program
stopwords.txt: a list of common stopwords to remove when generating the vocabulary
vocabulary.py: module that manages the update of vocabulary of documents

Input Format
============
The program expects two sets of files in the "input" directory:
1. <time_slice>.text: text of the documents, one line per document;
2. <time_slice>.time: time information of the documents, each line maps to the document that has the 
    same line number in <time_slice>.text

Parameter Settings
==================
Most parameter values (number of cores to use, minimum frequency threshold of vocabulary, etc) are 
set in lda.py. The number of topics can be specified in run_lda.sh. The size of the sliding window, 
however, is fixed at 2 time slices. Modification of the code is required to change this parameter.

Running the Program
===================
Execute the script run_lda.sh. It should process the sample input and generate the output in 
"output-<time_slice>" directories.
