import nltk, timeit
from nltk.corpus import treebank, brown, masc_tagged, conll2000
from nltk.tag import DefaultTagger
from nltk.tag import brill, brill_trainer
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger
from sklearn import metrics

# choose a train and test set
train_sents = treebank.tagged_sents()

textambig = nltk.word_tokenize("Time flies like an arrow. Time and flies are like an arrow. " #1.1/1.2
                               "We saw her duck with a saw. We saw her duck and it was small. " #2.1/2.2
                               "The complex houses married and single soldiers and their families. " #3.1
                               "The complex contains several houses. " #3.2
                               "The old man the boat. The old man sits in the boat. " #4.1/4.2
                               "The old train left the station. The old train the young men. " #5.1/5.2
                               "The trash can be smelly. The trash can is smelly. " #6.1/6.2
                               "The building blocks the sun. They are building blocks. There are building blocks. " #7.1/7.2/7.3
                               "That that exists in that area makes me nervous. " #8.1
                               "I know that this exists in the area of London. " #8.2
                               "The statue stands in the park. The statue stands in the park are rusty. " #9.1/9.2
                               "The cotton clothing is made of grows in Alabama. " #10.1
                               "The cotton that grows in Alabama is good. " #10.2
                               "The back door. The thing on my back. We win the voters back. " #11.1/11.2/11.3
                               "I promised to back the bill. " #11.4
                               "During its centennial year, The Wall Street Journal will report events of the past"
                               " century that stand as milestones of American business history. " #12.1
                               "We have a centennial this year. " #12.2 
                               "I have a round table. Yesterday we bought a round of cheese. " #13.1/13.2
                               "I think you should round out your interests. I have to work the year round. " #13.3/13.4
                               "She remembered everything in minute detail. She remembered that we meet in a minute. " #14.1/14.2
                               "She grabs her poles and skis down the mountain. " #15.1
                               "In winter she always skis down the mountain. " #15.2
                               "The insurance company receives a lot of calls and claims"
                               " that you submit your forms too late. " #16.1
                               "He had his claims back in 1756. " #16.2
                               "When the detective asks, the honest reply. " #17.1
                               "When the detective asks, the honest people reply. " #17.2
                               "I wonder if Will will sign his will. I have my own will. " #18.1/18.2
                               "Rose rose to put rose roes on her rose. Rose woke up to water all the red roses. " #19.1/19.2
                               "The flowers are rose. " #19.3
                               "Let the captain ship to the ship! The captain owns a ship. ") #20.1/20.2


textambigGoldStandard = [[('Time', 'NN'), ('flies', 'VBZ'), ('like', 'IN'), ('an', 'DT'), ('arrow', 'NN'), ('.', '.'),
                         ('Time', 'NN'), ('and', 'CC'), ('flies', 'NNS'), ('are', 'VBP'), ('like', 'IN'), ('an', 'DT'),
                         ('arrow', 'NN'), ('.', '.'), ('We', 'PRP'), ('saw', 'VBD'), ('her', 'PRP$'), ('duck', 'NN'),
                         ('with', 'IN'), ('a', 'DT'), ('saw', 'NN'), ('.', '.'), ('We', 'PRP'), ('saw', 'VBD'),
                         ('her', 'PRP$'), ('duck', 'NN'), ('and', 'CC'), ('it', 'PRP'), ('was', 'VBD'), ('small', 'JJ'),
                         ('.', '.'), ('The', 'DT'), ('complex', 'NN'), ('houses', 'VBZ'), ('married', 'JJ'),
                         ('and', 'CC'), ('single', 'JJ'), ('soldiers', 'NNS'), ('and', 'CC'), ('their', 'PRP$'),
                         ('families', 'NNS'), ('.', '.'), ('The', 'DT'), ('complex', 'NN'), ('contains', 'VBZ'),
                         ('several', 'JJ'), ('houses', 'NNS'), ('.', '.'), ('The', 'DT'), ('old', 'NNS'),
                         ('man', 'VBP'), ('the', 'DT'), ('boat', 'NN'), ('.', '.'), ('The', 'DT'), ('old', 'JJ'),
                         ('man', 'NN'), ('sits', 'VBZ'), ('in', 'IN'), ('the', 'DT'), ('boat', 'NN'), ('.', '.'),
                         ('The', 'DT'), ('old', 'JJ'), ('train', 'NN'), ('left', 'VBD'), ('the', 'DT'),
                         ('station', 'NN'), ('.', '.'), ('The', 'DT'), ('old', 'NNS'), ('train', 'VBP'), ('the', 'DT'),
                         ('young', 'JJ'), ('men', 'NNS'), ('.', '.'), ('The', 'DT'), ('trash', 'NN'), ('can', 'MD'),
                         ('be', 'VB'), ('smelly', 'RB'), ('.', '.'), ('The', 'DT'), ('trash', 'NN'), ('can', 'NN'),
                         ('is', 'VBZ'), ('smelly', 'RB'), ('.', '.'), ('The', 'DT'), ('building', 'NN'),
                         ('blocks', 'VBZ'), ('the', 'DT'), ('sun', 'NN'), ('.', '.'), ('They', 'PRP'), ('are', 'VBP'),
                         ('building', 'VBG'), ('blocks', 'NNS'), ('.', '.'), ('There', 'EX'), ('are', 'VBP'),
                         ('building', 'NN'), ('blocks', 'NNS'), ('.', '.'), ('That', 'IN'), ('that', 'DT'),
                         ('exists', 'VBZ'), ('in', 'IN'), ('that', 'DT'), ('area', 'NN'), ('makes', 'VBZ'),
                         ('me', 'PRP'), ('nervous', 'JJ'), ('.', '.'), ('I', 'PRP'), ('know', 'VBP'), ('that', 'IN'),
                         ('this', 'DT'), ('exists', 'VBZ'), ('in', 'IN'), ('the', 'DT'), ('area', 'NN'), ('of', 'IN'),
                         ('London', 'NNP'), ('.', '.'), ('The', 'DT'), ('statue', 'NN'), ('stands', 'VBZ'),
                         ('in', 'IN'), ('the', 'DT'), ('park', 'NN'), ('.', '.'), ('The', 'DT'), ('statue', 'NN'),
                         ('stands', 'NNS'), ('in', 'IN'), ('the', 'DT'), ('park', 'NN'), ('are', 'VBP'),
                         ('rusty', 'JJ'), ('.', '.'), ('The', 'DT'), ('cotton', 'NN'), ('clothing', 'NN'),
                         ('is', 'VBZ'), ('made', 'VBN'), ('of', 'IN'), ('grows', 'NNS'), ('in', 'IN'),
                         ('Alabama', 'NNP'), ('.', '.'), ('The', 'DT'), ('cotton', 'NN'), ('that', 'WDT'),
                         ('grows', 'VBZ'), ('in', 'IN'), ('Alabama', 'NNP'), ('is', 'VBZ'), ('good', 'JJ'), ('.', '.'),
                         ('The', 'DT'), ('back', 'NN'), ('door', 'NN'), ('.', '.'), ('The', 'DT'), ('thing', 'NN'),
                         ('on', 'IN'), ('my', 'PRP$'), ('back', 'NN'), ('.', '.'), ('We', 'PRP'), ('win', 'VBP'),
                         ('the', 'DT'), ('voters', 'NNS'), ('back', 'RB'), ('.', '.'), ('I', 'PRP'),
                         ('promised', 'VBD'), ('to', 'TO'), ('back', 'VB'), ('the', 'DT'), ('bill', 'NN'), ('.', '.'),
                         ('During', 'IN'), ('its', 'PRP$'), ('centennial', 'NN'), ('year', 'NN'), (',', ','),
                         ('The', 'NNP'), ('Wall', 'NNP'), ('Street', 'NNP'), ('Journal', 'NNP'), ('will', 'MD'),
                         ('report', 'VB'), ('events', 'NNS'), ('of', 'IN'), ('the', 'DT'), ('past', 'JJ'),
                         ('century', 'NN'), ('that', 'WDT'), ('stand', 'VBP'), ('as', 'IN'), ('milestones', 'NNS'),
                         ('of', 'IN'), ('American', 'JJ'), ('business', 'NN'), ('history', 'NN'), ('.', '.'),
                         ('We', 'PRP'), ('have', 'VBP'), ('a', 'DT'), ('centennial', 'NN'), ('this', 'DT'),
                         ('year', 'NN'), ('.', '.'), ('I', 'PRP'), ('have', 'VBP'), ('a', 'DT'), ('round', 'JJ'),
                         ('table', 'NN'), ('.', '.'), ('Yesterday', 'NN'), ('we', 'PRP'), ('bought', 'VBD'),
                         ('a', 'DT'), ('round', 'NN'), ('of', 'IN'), ('cheese', 'NN'), ('.', '.'), ('I', 'PRP'),
                         ('think', 'VBP'), ('you', 'PRP'), ('should', 'MD'), ('round', 'VB'), ('out', 'RP'),
                         ('your', 'PRP$'), ('interests', 'NNS'), ('.', '.'), ('I', 'PRP'), ('have', 'VBP'),
                         ('to', 'TO'), ('work', 'VB'), ('the', 'DT'), ('year', 'NN'), ('round', 'RB'), ('.', '.'),
                         ('She', 'PRP'), ('remembered', 'VBD'), ('everything', 'NN'), ('in', 'IN'), ('minute', 'JJ'),
                         ('detail', 'NN'), ('.', '.'), ('She', 'PRP'), ('remembered', 'VBD'), ('that', 'IN'),
                         ('we', 'PRP'), ('meet', 'VBP'), ('in', 'IN'), ('a', 'DT'), ('minute', 'NN'), ('.', '.'),
                         ('She', 'PRP'), ('grabs', 'VBZ'), ('her', 'PRP'), ('poles', 'NNS'), ('and', 'CC'),
                         ('skis', 'NNS'), ('down', 'RB'), ('the', 'DT'), ('mountain', 'NN'), ('.', '.'), ('In', 'IN'),
                         ('winter', 'NN'), ('she', 'PRP'), ('always', 'RB'), ('skis', 'VBZ'), ('down', 'RP'),
                         ('the', 'DT'), ('mountain', 'NN'), ('.', '.'), ('The', 'DT'), ('insurance', 'NN'),
                         ('company', 'NN'), ('receives', 'VBZ'), ('a', 'DT'), ('lot', 'NN'), ('of', 'IN'),
                         ('calls', 'NNS'), ('and', 'CC'), ('claims', 'VBZ'), ('that', 'IN'), ('you', 'PRP'),
                         ('submit', 'VBP'), ('your', 'PRP$'), ('forms', 'NNS'), ('too', 'RB'), ('late', 'RB'),
                         ('.', '.'), ('He', 'PRP'), ('had', 'VBD'), ('his', 'PRP$'), ('claims', 'NNS'), ('back', 'RB'),
                         ('in', 'IN'), ('1756', 'CD'), ('.', '.'), ('When', 'WRB'), ('the', 'DT'), ('detective', 'NN'),
                         ('asks', 'VBZ'), (',', ','), ('the', 'DT'), ('honest', 'NNS'), ('reply', 'VBP'), ('.', '.'),
                         ('When', 'WRB'), ('the', 'DT'), ('detective', 'NN'), ('asks', 'VBZ'), (',', ','),
                         ('the', 'DT'), ('honest', 'JJ'), ('people', 'NNS'), ('reply', 'VBP'), ('.', '.'),
                         ('I', 'PRP'), ('wonder', 'VBP'), ('if', 'IN'), ('Will', 'NNP'), ('will', 'MD'), ('sign', 'VB'),
                         ('his', 'PRP$'), ('will', 'NN'), ('.', '.'), ('I', 'PRP'), ('have', 'VBP'), ('my', 'PRP$'),
                         ('own', 'JJ'), ('will', 'NN'), ('.', '.'), ('Rose', 'NN'), ('rose', 'VBD'), ('to', 'TO'),
                         ('put', 'VB'), ('rose', 'JJ'), ('roes', 'NNS'), ('on', 'IN'), ('her', 'PRP$'),
                         ('rose', 'NN'), ('.', '.'), ('Rose', 'NNP'), ('woke', 'VBD'), ('up', 'RP'), ('to', 'TO'),
                         ('water', 'VB'), ('all', 'PDT'), ('the', 'DT'), ('red', 'JJ'), ('roses', 'NNS'), ('.', '.'),
                         ('The', 'DT'), ('flowers', 'NNS'), ('are', 'VBP'), ('rose', 'JJ'), ('.', '.'), ('Let', 'VB'),
                         ('the', 'DT'), ('captain', 'NN'), ('ship', 'VB'), ('to', 'TO'), ('the', 'DT'), ('ship', 'NN'),
                         ('!', '.'), ('The', 'DT'), ('captain', 'NN'), ('owns', 'VBZ'), ('a', 'DT'),
                         ('ship', 'NN'), ('.', '.')]]


textambigGoldStandardBrown = [[('Time', 'NN'), ('flies', 'VBZ'), ('like', 'IN'), ('an', 'AT'), ('arrow', 'NN'),
                               ('.', '.'), ('Time', 'NN'), ('and', 'CC'), ('flies', 'NNS'), ('are', 'BER'),
                               ('like', 'IN'), ('an', 'AT'), ('arrow', 'NN'), ('.', '.'), ('We', 'PPSS'),
                               ('saw', 'VBD'), ('her', 'PRP$'), ('duck', 'NN'), ('with', 'IN'), ('a', 'AT'),
                               ('saw', 'NN'), ('.', '.'), ('We', 'PPSS'), ('saw', 'VBD'), ('her', 'PRP$'),
                               ('duck', 'NN'), ('and', 'CC'), ('it', 'PPS'), ('was', 'BEDZ'), ('small', 'JJ'),
                               ('.', '.'), ('The', 'AT'), ('complex', 'NN'), ('houses', 'VBZ'), ('married', 'JJ'),
                               ('and', 'CC'), ('single', 'JJ'), ('soldiers', 'NNS'), ('and', 'CC'), ('their', 'PP$'),
                               ('families', 'NNS'), ('.', '.'), ('The', 'AT'), ('complex', 'NN'), ('contains', 'VBZ'),
                               ('several', 'AP'), ('houses', 'NNS'), ('.', '.'), ('The', 'AT'), ('old', 'NNS'),
                               ('man', 'VBP'), ('the', 'AT'), ('boat', 'NN'), ('.', '.'), ('The', 'AT'), ('old', 'JJ'),
                               ('man', 'NN'), ('sits', 'VBZ'), ('in', 'IN'), ('the', 'AT'), ('boat', 'NN'), ('.', '.'),
                               ('The', 'AT'), ('old', 'JJ'), ('train', 'NN'), ('left', 'VBD'), ('the', 'AT'),
                               ('station', 'NN'), ('.', '.'), ('The', 'AT'), ('old', 'NNS'), ('train', 'VBP'),
                               ('the', 'AT'), ('young', 'JJ'), ('men', 'NNS'), ('.', '.'), ('The', 'AT'),
                               ('trash', 'NN'), ('can', 'MD'), ('be', 'BE'), ('smelly', 'RB'), ('.', '.'),
                               ('The', 'AT'), ('trash', 'NN'), ('can', 'NN'), ('is', 'BEZ'), ('smelly', 'RB'),
                               ('.', '.'), ('The', 'AT'), ('building', 'NN'), ('blocks', 'VBZ'), ('the', 'AT'),
                               ('sun', 'NN'), ('.', '.'), ('They', 'PPSS'), ('are', 'BER'), ('building', 'VBG'),
                               ('blocks', 'NNS'), ('.', '.'), ('There', 'EX'), ('are', 'BER'), ('building', 'NN'),
                               ('blocks', 'NNS'), ('.', '.'), ('That', 'CS'), ('that', 'DT'), ('exists', 'VBZ'),
                               ('in', 'IN'), ('that', 'DT'), ('area', 'NN'), ('makes', 'VBZ'), ('me', 'PPO'),
                               ('nervous', 'JJ'), ('.', '.'), ('I', 'PPSS'), ('know', 'VB'), ('that', 'CS'),
                               ('this', 'DT'), ('exists', 'VBZ'), ('in', 'IN'), ('the', 'AT'), ('area', 'NN'),
                               ('of', 'IN'), ('London', 'NP'), ('.', '.'), ('The', 'AT'), ('statue', 'NN'),
                               ('stands', 'VBZ'), ('in', 'IN'), ('the', 'AT'), ('park', 'NN'), ('.', '.'),
                               ('The', 'AT'), ('statue', 'NN'), ('stands', 'NNS'), ('in', 'IN'), ('the', 'AT'),
                               ('park', 'NN'), ('are', 'BER'), ('rusty', 'JJ'), ('.', '.'), ('The', 'AT'),
                               ('cotton', 'NN'), ('clothing', 'NN'), ('is', 'BEZ'), ('made', 'VBN'), ('of', 'IN'),
                               ('grows', 'NNS'), ('in', 'IN'), ('Alabama', 'NP'), ('.', '.'), ('The', 'AT'),
                               ('cotton', 'NN'), ('that', 'WPS'), ('grows', 'VBZ'), ('in', 'IN'), ('Alabama', 'NP'),
                               ('is', 'BEZ'), ('good', 'JJ'), ('.', '.'), ('The', 'AT'), ('back', 'NN'), ('door', 'NN'),
                               ('.', '.'), ('The', 'AT'), ('thing', 'NN'), ('on', 'IN'), ('my', 'PP$'), ('back', 'NN'),
                               ('.', '.'), ('We', 'PPSS'), ('win', 'VBP'), ('the', 'AT'), ('voters', 'NNS'),
                               ('back', 'RB'), ('.', '.'), ('I', 'PPSS'), ('promised', 'VBD'), ('to', 'TO'),
                               ('back', 'VB'), ('the', 'AT'), ('bill', 'NN'), ('.', '.'), ('During', 'IN'),
                               ('its', 'PP$'), ('centennial', 'NN'), ('year', 'NN'), (',', ','), ('The', 'AT-TL'),
                               ('Wall', 'NP-TL'), ('Street', 'NP-TL'), ('Journal', 'NP-TL'), ('will', 'MD'),
                               ('report', 'VB'), ('events', 'NNS'), ('of', 'IN'), ('the', 'AT'), ('past', 'AP'),
                               ('century', 'NN'), ('that', 'WPS'), ('stand', 'VBP'), ('as', 'CS'), ('milestones', 'NNS'),
                               ('of', 'IN'), ('American', 'JJ'), ('business', 'NN'), ('history', 'NN'), ('.', '.'),
                               ('We', 'PPSS'), ('have', 'HV'), ('a', 'AT'), ('centennial', 'NN'), ('this', 'DT'),
                               ('year', 'NN'), ('.', '.'), ('I', 'PPSS'), ('have', 'HV'), ('a', 'AT'), ('round', 'JJ'),
                               ('table', 'NN'), ('.', '.'), ('Yesterday', 'NR'), ('we', 'PPSS'), ('bought', 'VBD'),
                               ('a', 'AT'), ('round', 'NN'), ('of', 'IN'), ('cheese', 'NN'), ('.', '.'), ('I', 'PPSS'),
                               ('think', 'VBP'), ('you', 'PPSS'), ('should', 'MD'), ('round', 'VB'), ('out', 'RP'),
                               ('your', 'PP$'), ('interests', 'NNS'), ('.', '.'), ('I', 'PPSS'), ('have', 'HV'),
                               ('to', 'TO'), ('work', 'VB'), ('the', 'AT'), ('year', 'NN'), ('round', 'RB'), ('.', '.'),
                               ('She', 'PPS'), ('remembered', 'VBD'), ('everything', 'PN'), ('in', 'IN'),
                               ('minute', 'JJ'), ('detail', 'NN'), ('.', '.'), ('She', 'PPS'), ('remembered', 'VBD'),
                               ('that', 'CS'), ('we', 'PPSS'), ('meet', 'VBP'), ('in', 'IN'), ('a', 'AT'),
                               ('minute', 'NN'), ('.', '.'), ('She', 'PPS'), ('grabs', 'VBZ'), ('her', 'PP$'),
                               ('poles', 'NNS'), ('and', 'CC'), ('skis', 'NNS'), ('down', 'RB'), ('the', 'AT'),
                               ('mountain', 'NN'), ('.', '.'), ('In', 'IN'), ('winter', 'NN'), ('she', 'PPS'),
                               ('always', 'RB'), ('skis', 'VBZ'), ('down', 'RP'), ('the', 'AT'), ('mountain', 'NN'),
                               ('.', '.'), ('The', 'AT'), ('insurance', 'NN'), ('company', 'NN'), ('receives', 'VBZ'),
                               ('a', 'AT'), ('lot', 'NN'), ('of', 'IN'), ('calls', 'NNS'), ('and', 'CC'),
                               ('claims', 'VBZ'), ('that', 'CS'), ('you', 'PPSS'), ('submit', 'VBP'), ('your', 'PP$'),
                               ('forms', 'NNS'), ('too', 'QL'), ('late', 'RB'), ('.', '.'), ('He', 'PPS'),
                               ('had', 'HVD'), ('his', 'PP$'), ('claims', 'NNS'), ('back', 'RB'), ('in', 'IN'),
                               ('1756', 'CD'), ('.', '.'), ('When', 'WRB'), ('the', 'AT'), ('detective', 'NN'),
                               ('asks', 'NN'), (',', ','), ('the', 'AT'), ('honest', 'NNS'), ('reply', 'VBP'),
                               ('.', '.'), ('When', 'WRB'), ('the', 'AT'), ('detective', 'NN'), ('asks', 'NN'),
                               (',', ','), ('the', 'AT'), ('honest', 'JJ'), ('people', 'NNS'), ('reply', 'VBP'),
                               ('.', '.'), ('I', 'PPSS'), ('wonder', 'VBP'), ('if', 'CS'), ('Will', 'NP'),
                               ('will', 'MD'), ('sign', 'VB'), ('his', 'PP$'), ('will', 'NN'), ('.', '.'),
                               ('I', 'PPSS'), ('have', 'HV'), ('my', 'PP$'), ('own', 'JJ'), ('will', 'NN'), ('.', '.'),
                               ('Rose', 'NP'), ('rose', 'VBD'), ('to', 'TO'), ('put', 'VB'), ('rose', 'JJ'),
                               ('roes', 'NNS'), ('on', 'IN'), ('her', 'PP$'), ('rose', 'NN'), ('.', '.'),
                               ('Rose', 'NP'), ('woke', 'VBD'), ('up', 'RP'), ('to', 'TO'), ('water', 'VB'),
                               ('all', 'ABN'), ('the', 'AT'), ('red', 'JJ'), ('roses', 'NNS'), ('.', '.'),
                               ('The', 'AT'), ('flowers', 'NNS'), ('are', 'BER'), ('rose', 'JJ'), ('.', '.'),
                               ('Let', 'VB'), ('the', 'AT'), ('captain', 'NN'), ('ship', 'VB'), ('to', 'IN'),
                               ('the', 'AT'), ('ship', 'NN'), ('!', '.'), ('The', 'AT'), ('captain', 'NN'),
                               ('owns', 'VBZ'), ('a', 'AT'), ('ship', 'NN'), ('.', '.')]]

textnewwords = nltk.word_tokenize("I like to wear my mankini. " #noun #1
                                  "Yesterday I bought a lot of bling. " #noun #2
                                  "This jewellery is very droolworthy. " #adj #3
                                  "I forgot which state is a Purple State. " #noun #4
                                  "I androgynously matched you wrong. " #adv #5
                                  "This society is degendered. " #adj #6
                                  "You envenomate your father. " #verb vergiften #7
                                  "This is an eventitive event. " #adj #8
                                  "This is an eventitive. "#noun #9
                                  "Everwhen I go to the city this happens. "#conj #10
                                  "I will jackhammer this building. " #11
                                  "I jackhammer this building. " #12
                                  "These are unjournalistic methods. " #adj #13
                                  "The unknot arises in the mathematical theory of knots. "#noun #14
                                  "The virus can then spread to new individuals through skin-to-skin contact .") #adj #15

textnewwordsGoldStandard = [[('I', 'PRP'), ('like', 'VBP'), ('to', 'TO'), ('wear', 'VB'), ('my', 'PRP$'),
                            ('mankini', 'NN'), ('.', '.'), ('Yesterday', 'RB'), ('I', 'PRP'), ('bought', 'VBD'),
                            ('a', 'DT'), ('lot', 'NN'), ('of', 'IN'), ('bling', 'NN'), ('.', '.'), ('This', 'DT'),
                            ('jewellery', 'NN'), ('is', 'VBZ'), ('very', 'RB'), ('droolworthy', 'JJ'), ('.', '.'),
                            ('I', 'PRP'), ('forgot', 'VBD'), ('which', 'WDT'), ('state', 'NN'), ('is', 'VBZ'),
                            ('a', 'DT'), ('Purple', 'NNP'), ('State', 'NNP'), ('.', '.'), ('I', 'PRP'),
                            ('androgynously', 'RB'), ('matched', 'VBD'), ('you', 'PRP'), ('wrong', 'JJ'), ('.', '.'),
                            ('This', 'DT'), ('society', 'NN'), ('is', 'VBZ'), ('degendered', 'JJ'), ('.', '.'),
                            ('You', 'PRP'), ('envenomate', 'VBP'), ('your', 'PRP$'), ('father', 'NN'), ('.', '.'),
                            ('This', 'DT'), ('is', 'VBZ'), ('an', 'DT'), ('eventitive', 'JJ'), ('event', 'NN'),
                            ('.', '.'), ('This', 'DT'), ('is', 'VBZ'), ('an', 'DT'), ('eventitive', 'NN'), ('.', '.'),
                            ('Everwhen', 'IN'), ('I', 'PRP'), ('go', 'VBP'), ('to', 'TO'), ('the', 'DT'),
                            ('city', 'NN'), ('this', 'DT'), ('happens', 'VBZ'), ('.', '.'), ('I', 'PRP'),
                            ('will', 'MD'), ('jackhammer', 'VB'), ('this', 'DT'), ('building', 'NN'), ('.', '.'),
                            ('I', 'PRP'), ('jackhammer', 'VBP'), ('this', 'DT'), ('building', 'NN'), ('.', '.'),
                            ('These', 'DT'), ('are', 'VBP'), ('unjournalistic', 'JJ'), ('methods', 'NNS'), ('.', '.'),
                            ('The', 'DT'), ('unknot', 'NN'), ('arises', 'VBZ'), ('in', 'IN'), ('the', 'DT'),
                            ('mathematical', 'JJ'), ('theory', 'NN'), ('of', 'IN'), ('knots', 'NNS'), ('.', '.'),
                            ('The', 'DT'), ('virus', 'NN'), ('can', 'MD'), ('then', 'RB'), ('spread', 'VB'),
                            ('to', 'TO'), ('new', 'JJ'), ('individuals', 'NNS'), ('through', 'IN'),
                            ('skin-to-skin', 'JJ'), ('contact', 'NN'), ('.', '.')]]

textnewwordsGoldStandardBrown = [[('I', 'PPSS'), ('like', 'VBP'), ('to', 'TO'), ('wear', 'VB'), ('my', 'PP$'),
                                  ('mankini', 'NN'), ('.', '.'), ('Yesterday', 'NR'), ('I', 'PPSS'), ('bought', 'VBD'),
                                  ('a', 'AT'), ('lot', 'NN'), ('of', 'IN'), ('bling', 'NN'), ('.', '.'), ('This', 'DT'),
                                  ('jewellery', 'NN'), ('is', 'BEZ'), ('very', 'QL'), ('droolworthy', 'JJ'), ('.', '.'),
                                  ('I', 'PPSS'), ('forgot', 'VBD'), ('which', 'WDT'), ('state', 'NN'), ('is', 'BEZ'),
                                  ('a', 'AT'), ('Purple', 'NP'), ('State', 'NP'), ('.', '.'), ('I', 'PPSS'),
                                  ('androgynously', 'RB'), ('matched', 'VBD'), ('you', 'PPO'), ('wrong', 'JJ'),
                                  ('.', '.'), ('This', 'DT'), ('society', 'NN'), ('is', 'BEZ'), ('degendered', 'JJ'),
                                  ('.', '.'), ('You', 'PPSS'), ('envenomate', 'VBP'), ('your', 'PP$'), ('father', 'NN'),
                                  ('.', '.'), ('This', 'DT'), ('is', 'BEZ'), ('an', 'AT'), ('eventitive', 'JJ'),
                                  ('event', 'NN'), ('.', '.'), ('This', 'DT'), ('is', 'BEZ'), ('an', 'AT'),
                                  ('eventitive', 'NN'), ('.', '.'), ('Everwhen', 'CS'), ('I', 'PPSS'), ('go', 'VBP'),
                                  ('to', 'IN'), ('the', 'AT'), ('city', 'NN'), ('this', 'DT'), ('happens', 'VBZ'),
                                  ('.', '.'), ('I', 'PPSS'), ('will', 'MD'), ('jackhammer', 'VB'), ('this', 'DT'),
                                  ('building', 'NN'), ('.', '.'), ('I', 'PPSS'), ('jackhammer', 'VBP'), ('this', 'DT'),
                                  ('building', 'NN'), ('.', '.'), ('These', 'DTS'), ('are', 'BER'),
                                  ('unjournalistic', 'JJ'), ('methods', 'NNS'), ('.', '.'), ('The', 'AT'),
                                  ('unknot', 'NN'), ('arises', 'VBZ'), ('in', 'IN'), ('the', 'AT'),
                                  ('mathematical', 'JJ'), ('theory', 'NN'), ('of', 'IN'), ('knots', 'NNS'),
                                  ('.', '.'), ('The', 'AT'), ('virus', 'NN'), ('can', 'MD'), ('then', 'RB'),
                                  ('spread', 'VB'), ('to', 'IN'), ('new', 'JJ'), ('individuals', 'NNS'),
                                  ('through', 'IN'), ('skin-to-skin', 'JJ'), ('contact', 'NN'), ('.', '.')]]

# create a brill tagger function
def train_brill_tagger(initial_tagger, train_sents, **kwargs):
    templates = [
        brill.Template(brill.Pos([-1])),
        brill.Template(brill.Pos([1])),
        brill.Template(brill.Pos([-2])),
        brill.Template(brill.Pos([2])),
        brill.Template(brill.Pos([-2, -1])),
        brill.Template(brill.Pos([1, 2])),
        brill.Template(brill.Pos([-3, -2, -1])),
        brill.Template(brill.Pos([1, 2, 3])),
        brill.Template(brill.Pos([-1]), brill.Pos([1])),
        brill.Template(brill.Word([-1])),
        brill.Template(brill.Word([1])),
        brill.Template(brill.Word([-2])),
        brill.Template(brill.Word([2])),
        brill.Template(brill.Word([-2, -1])),
        brill.Template(brill.Word([1, 2])),
        brill.Template(brill.Word([-3, -2, -1])),
        brill.Template(brill.Word([1, 2, 3])),
        brill.Template(brill.Word([-1]), brill.Word([1])),
        ]
    trainer = brill_trainer.BrillTaggerTrainer(initial_tagger,
     templates, deterministic=True)
    return trainer.train(train_sents, **kwargs)

# create an initial tagger
def backoff_tagger(train_sents, tagger_classes, backoff=None):
    for cls in tagger_classes:
        backoff = cls(train_sents, backoff=backoff)
        return backoff

# create backoff tagger
default_tagger = DefaultTagger('NN')
tagger = UnigramTagger(train_sents)

initial_tagger = backoff_tagger(train_sents, [UnigramTagger,BigramTagger, TrigramTagger], backoff=default_tagger)

def tagger1():
    train_sents = treebank.tagged_sents()
    def train():
        initial_tagger = backoff_tagger(train_sents, [UnigramTagger, BigramTagger, TrigramTagger],
                                        backoff=default_tagger)
    print("Time to train: ", timeit.timeit(train, number=1))
    def tag():
        initial_tagger.tag(textambig)
    print("Time to tag ambiguous: ",timeit.timeit(tag, number=1))
    def tag2():
        initial_tagger.tag(textnewwords)
    print("Time to tag unknown: ",timeit.timeit(tag2, number=1))
    print("")
    print('Evaluation inital tagger, treebank, ambiguous words:')
    print("")
    print(initial_tagger.tag(textambig))
    print("")
    print('accuracy: ', initial_tagger.evaluate(textambigGoldStandard))
    gold = [str(tag) for sentence in textambigGoldStandard for token, tag in sentence]
    pred = [str(tag) for sentence in zip(initial_tagger.tag(textambig)) for token, tag in sentence]
    print(metrics.classification_report(gold, pred))
    print("")
    print('Evaluation initial, treebank, unknown words:')
    print("")
    print(initial_tagger.tag(textnewwords))
    print("")
    print('accuracy: ', initial_tagger.evaluate(textnewwordsGoldStandard))
    gold = [str(tag) for sentence in textnewwordsGoldStandard for token, tag in sentence]
    pred = [str(tag) for sentence in zip(initial_tagger.tag(textnewwords)) for token, tag in sentence]
    print(metrics.classification_report(gold, pred))
    print("")

def tagger2():
    train_sents2 = treebank.tagged_sents()
    initial_tagger2 = backoff_tagger(train_sents2, [UnigramTagger, BigramTagger, TrigramTagger], backoff=default_tagger)
    brill_tagger = train_brill_tagger(initial_tagger2, train_sents2)
    def train():
        brill_tagger = train_brill_tagger(initial_tagger2, train_sents2)
    print("Time to train: ", timeit.timeit(train, number=1))
    def tag():
        brill_tagger.tag(textambig)
    print("Time to tag ambiguous: ",timeit.timeit(tag, number=1))
    def tag2():
        brill_tagger.tag(textnewwords)
    print("Time to tag unknown: ",timeit.timeit(tag2, number=1))
    print("")
    print('Evaluation BrillTagger, treebank, ambiguous words:')
    print("")
    print(brill_tagger.tag(textambig))
    print("")
    print('accuracy: ', brill_tagger.evaluate(textambigGoldStandard))
    gold = [str(tag) for sentence in textambigGoldStandard for token, tag in sentence]
    pred = [str(tag) for sentence in zip(brill_tagger.tag(textambig)) for token, tag in sentence]
    print(metrics.classification_report(gold, pred))
    print("")
    print('Evaluation BrillTagger, treebank, unknown words:')
    print("")
    print(brill_tagger.tag(textnewwords))
    print("")
    print('accuracy: ', brill_tagger.evaluate(textnewwordsGoldStandard))
    gold = [str(tag) for sentence in textnewwordsGoldStandard for token, tag in sentence]
    pred = [str(tag) for sentence in zip(brill_tagger.tag(textnewwords)) for token, tag in sentence]
    print(metrics.classification_report(gold, pred))
    print("")

def tagger3():
    train_sents3 = brown.tagged_sents()
    initial_tagger3 = backoff_tagger(train_sents3, [UnigramTagger, BigramTagger, TrigramTagger], backoff=default_tagger)
    brill_tagger = train_brill_tagger(initial_tagger3, train_sents3)
    def train():
        brill_tagger = train_brill_tagger(initial_tagger3, train_sents3)
    print("Time to train: ", timeit.timeit(train, number=1))
    def tag():
        brill_tagger.tag(textambig)
    print("Time to tag ambiguous: ",timeit.timeit(tag, number=1))
    def tag2():
        brill_tagger.tag(textnewwords)
    print("Time to tag unknown: ",timeit.timeit(tag2, number=1))
    print("")
    print('Evaluation BrillTagger, brown, ambiguous words:')
    print("")
    print(brill_tagger.tag(textambig))
    print("")
    print('accuracy: ', brill_tagger.evaluate(textambigGoldStandardBrown))
    gold = [str(tag) for sentence in textambigGoldStandardBrown for token, tag in sentence]
    pred = [str(tag) for sentence in zip(brill_tagger.tag(textambig)) for token, tag in sentence]
    print(metrics.classification_report(gold, pred))
    print("")
    print('Evaluation initial, brown, unknown words:')
    print("")
    print(brill_tagger.tag(textnewwords))
    print("")
    print('accuracy: ', brill_tagger.evaluate(textnewwordsGoldStandardBrown))
    gold = [str(tag) for sentence in textnewwordsGoldStandardBrown for token, tag in sentence]
    pred = [str(tag) for sentence in zip(brill_tagger.tag(textnewwords)) for token, tag in sentence]
    print(metrics.classification_report(gold, pred))
    print("")

def tagger4():
    train_sents4 = conll2000.tagged_sents()
    initial_tagger4 = backoff_tagger(train_sents4, [UnigramTagger, BigramTagger, TrigramTagger], backoff=default_tagger)
    brill_tagger = train_brill_tagger(initial_tagger4, train_sents4)
    def train():
        brill_tagger = train_brill_tagger(initial_tagger4, train_sents4)
    print("Time to train: ", timeit.timeit(train, number=1))
    def tag():
        brill_tagger.tag(textambig)
    print("Time to tag ambiguous: ",timeit.timeit(tag, number=1))
    def tag2():
        brill_tagger.tag(textnewwords)
    print("Time to tag unknown: ",timeit.timeit(tag2, number=1))
    print("")
    print('Evaluation BrillTagger, conll, ambiguous words:')
    print("")
    print(brill_tagger.tag(textambig))
    print("")
    print('accuracy: ', brill_tagger.evaluate(textambigGoldStandard))
    gold = [str(tag) for sentence in textambigGoldStandard for token, tag in sentence]
    pred = [str(tag) for sentence in zip(brill_tagger.tag(textambig)) for token, tag in sentence]
    print(metrics.classification_report(gold, pred))
    print("")
    print('Evaluation initial, conll, unknown words:')
    print("")
    print(brill_tagger.tag(textnewwords))
    print("")
    print('accuracy: ', brill_tagger.evaluate(textnewwordsGoldStandard))
    gold = [str(tag) for sentence in textnewwordsGoldStandard for token, tag in sentence]
    pred = [str(tag) for sentence in zip(brill_tagger.tag(textnewwords)) for token, tag in sentence]
    print(metrics.classification_report(gold, pred))
    print("")

def tagger5():
    train_sents5 = masc_tagged.tagged_sents()
    initial_tagger5 = backoff_tagger(train_sents5, [UnigramTagger, BigramTagger, TrigramTagger], backoff=default_tagger)
    brill_tagger = train_brill_tagger(initial_tagger5, train_sents5)
    def train():
        brill_tagger = train_brill_tagger(initial_tagger5, train_sents5)
    print("Time to train: ", timeit.timeit(train, number=1))
    def tag():
        brill_tagger.tag(textambig)
    print("Time to tag ambiguous: ",timeit.timeit(tag, number=1))
    def tag2():
        brill_tagger.tag(textnewwords)
    print("Time to tag unknown: ",timeit.timeit(tag2, number=1))
    print("")
    print('Evaluation BrillTagger, masc, ambiguous words:')
    print("")
    print(brill_tagger.tag(textambig))
    print("")
    print('accuracy: ', brill_tagger.evaluate(textambigGoldStandard))
    gold = [str(tag) for sentence in textambigGoldStandard for token, tag in sentence]
    pred = [str(tag) for sentence in zip(brill_tagger.tag(textambig)) for token, tag in sentence]
    print(metrics.classification_report(gold, pred))
    print("")
    print('Evaluation initial, masc, unknown words:')
    print("")
    print(brill_tagger.tag(textnewwords))
    print("")
    print('accuracy: ', brill_tagger.evaluate(textnewwordsGoldStandard))
    gold = [str(tag) for sentence in textnewwordsGoldStandard for token, tag in sentence]
    pred = [str(tag) for sentence in zip(brill_tagger.tag(textnewwords)) for token, tag in sentence]
    print(metrics.classification_report(gold, pred))
    print("")

tagger1()
tagger2()
tagger3()
tagger4()
tagger5()