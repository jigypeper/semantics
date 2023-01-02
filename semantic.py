"""
This is a notebook with some observations on Semantic Similarity
Note: for this to run you will need to install spacy and download the language model
'pip install spacy'
'python3 -m spacy download en_core_web_md'
"""

import spacy

# load the language models
nlp = spacy.load("en_core_web_md")
nlp_simple = spacy.load("en_core_web_sm")

# define some words to compare and pass them to the model
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
word4 = nlp("lion")
word5 = nlp("leopard")

# display the similarity result (ranges between 0 and 1, 1 being most similar)
# cat vs monkey, these are similar because both are animals
print(word1.similarity(word2))

# banana vs monkey, although these are not animals, there is a relationship because monkeys eat bananas
print(word3.similarity(word2))

# banana vs cat, this comparison has the least similarity, because cats are carnivores, and generally
# are not linked semantically with fruits
print(word3.similarity(word1))

# cat vs lion, I would expect this to have a higher similarity than cat and monkey, but oddly, it doesn't
print(word1.similarity(word4))

# lion vs leopard, this is as expected because both are large felines
print(word4.similarity(word5))

# looping through words in a sentence and comparing. For this loop, a very interesting comparison is that
# for lion and monkey, the similarity of the 2 (0.49) appears to be higher than that of cat and lion (0.38), even though
# both cat and lion are felines. This may be related to more factors like where they are likely to be found
# and other aspects like domestication.
tokens = nlp('cat apple monkey banana lion')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# simple model test
word1_simple = nlp_simple("cat")
word2_simple = nlp_simple("monkey")

# cat vs monkey with simple model. The similarity is much greater because the model does not have vectors.
# The comparison is likely based on the tags in this case, in reality a cat and monkey do not
# have a 74% semantic similarity
print(word1_simple.similarity(word2_simple))

# Define sentence to compare, pass it to model
sentence_to_compare = "Why is my cat on the car"
model_sentence = nlp(sentence_to_compare)

# define sentences to compare against
sentences = [
    "where did my dog go",
    "Hello, there is my car",
    "I\'ve lost my car in my car",
    "I\'d like my boat back",
    "I will name my dog Diana",
    "My cat is afraid of cars"
]

# loop through sentences and compare, display the similarity.
# the most similar sentence is the first, this is likely because both sentences refer to cars
# that belong to the user. Adding a sentence in the list ascertains my thoughts on
# this, the sentence includes both cats and cars yet only has the same similarity as the completely
# unrelated sentence "I will name my dog Diana"
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)
