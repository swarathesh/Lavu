''' Make LIWC feature extractor into class
'''
import nltk
import re
import pickle
liwcPath = 'liwc/liwc-english-mod.dic'

def makeLIWCDictionary(liwcPath, picklePath):
    '''
        Make lookup data structure from LIWC dictionary file
    '''
    LIWC_file = open(liwcPath, 'rb') # LIWC dictionary
    catNames = {}
    LIWC_file.readline() #skips first '%' line
    line = LIWC_file.readline()
    lookup = []
    while '%' not in line:
        keyval = line.split('\t')
        key = keyval[0]
        value = keyval[1].strip()
        catNames[key] = {'name' : value,
                         'words' : []}
        line = LIWC_file.readline()
    mapCategoriesToNumbers = catNames.keys()
    line = LIWC_file.readline() # skips second '%' line

    #return mapCategoriesToNumbers
    while line: #iterate through categories
        data = line.strip().split('\t')
        reString = '^'+data[0].replace('*', '.*') + '$'
        indeces = [mapCategoriesToNumbers.index(d) for d in data[1:]]
        lookupCell = (re.compile(reString), indeces)
        lookup.append(lookupCell)
        for cat in data[1:]:
            catNames[cat]['words'] += (data[0], reString)
        cats = data[1:]
        line = LIWC_file.readline()
    toPickle = {'categories' : catNames, 'lookup' : lookup, 'cat_to_num' : mapCategoriesToNumbers}
    pickle.dump(toPickle, open(picklePath, 'w'))
    return toPickle

class liwcExtractor():
    def __init__(self,
                tokenizer=None,
                ignore=None,
                dictionary=None,
                newCategories=None,
                keepNonDict=True,
                liwcPath=None):
        self.liwcPath = liwcPath
        self.dictionary = dictionary
        if tokenizer is None:
            self.tokenizer = self.nltk_tokenize
        if liwcPath is not None:
            self.dictionary = makeLIWCDictionary(liwcPath, './liwcDictionary.pickle')
            self.lookup = self.dictionary['lookup']
            self.categories = self.dictionary['categories']
            self.mapCategoriesToNumbers = self.dictionary['cat_to_num']
        elif self.dictionary==None:
            self.dictionary = makeLIWCDictionary(liwcPath, './liwcDictionary.pickle')
            self.lookup = self.dictionary['lookup']
            self.categories = self.dictionary['categories']
            self.mapCategoriesToNumbers = self.dictionary['cat_to_num']
        self.ignore = ignore
        self.newCategories = newCategories
        self.nonDictTokens = []
        self.keepNonDict = keepNonDict

    def getCategoryIndeces(self):
        indeces = [x['name'] for x in self.categories.values()]
        indeces += ['wc', 'sixltr','dic','punc','emoticon'] # These last two are not built yet.
        return indeces

    def extract(self, corpus):
        corpusFeatures = []
        for doc in corpus:
            features = self.extractFromDoc(doc)
            corpusFeatures.append(features)
        return corpusFeatures

    def extractFromDoc(self, document):
        tokens = self.tokenizer(document)
        #print tokens
        features = [0] * 70 # 66 = wc, total word count
                            # 67 = sixltr, six letter words
                            # 68 = dic, words found in LIWC dictionary
                            # 70 = punc, punctuation
                            # 71 = emoticon
        features[66] = len(tokens)

        for t in tokens: #iterating through tokens of a message
            #print "Token : " + t
            if len(t) > 6: # check if more than six letters
                features[67] += 1
            inDict = False
            for pattern, categories in self.lookup:
                if len(pattern.findall(t)) > 0:
                    inDict = True
                    for c in categories:
                        features[int(c)] += 1
            if inDict:
                features[68] += 1
            else:
                self.nonDictTokens.append(t)
        return features

    def patternsMatchedFromDoc(self, document):
        tokens = self.tokenizer(document)
        patterns = [l[0] for l in self.lookup]
        features = [0] * len(patterns)
        for t in tokens:
            for i, pattern in enumerate(patterns):
                if len(pattern.findall(t)) > 0:
                    features[i] += 1
        return features

    def nltk_tokenize(self, message):
        '''
            takes in a text string and returns a list of tokenized words using nltk methods
        '''
        # sentence tokenize
        stList = nltk.sent_tokenize(message)
        # word tokenize
        tokens = []
        for sent in stList:
            tokens += nltk.word_tokenize(sent)
        return tokens