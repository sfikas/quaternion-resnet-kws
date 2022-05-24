'''
Based on code by ssudholdt
@author: sfikas
'''


import logging
import numpy as np
import tqdm
import unittest
import itertools
from .ngrams import get_most_common_n_grams
from .phoclayers import PhocLayerPart

class PhocLayout():
    possible_unigrams = []
    possible_bigrams = []
    unigram_levels = []
    bigram_levels = []

    def __init__(self, 
        possible_unigrams=[], 
        unigram_levels=[], 
        possible_bigrams=[], 
        bigram_levels=[],
        use_all_unigram_combinations_as_bigrams=False,      # Default is the 75 bigrams of Almazan; or else, use all combinations
        use_agnostic_bigram=True,                           # Add a bigram that is to be used as a catch-all bigram for bigrams not included explicitly
        zero_layers_over_wordlength=False,                  # All layers that are finer than the word length will be zeroed
        occupation_threshold=0.4,                           #This is used to determine if an ngram is part of a PHOC part (Sudholt code sets it to 0.5)
        unibi_consistency_penalty=(1e-1, 1., 1e4, 1e4),     #These control the penalty function for unigram-bigram inconsistency (see compute_unigrambigram_cliquematrix and paper for details)
        confidence_unigrams=(1., 0.),                       #Confidence over unigram/bigram layers = (base weight, exponential decay*). 
        confidence_bigrams=(1., 0.),                        #*the decay is over the level of the layer, so higher level = more decay = less confidence.
        agnostic_bigram_penalty=1.,                         #1.: No penalty. -> 0: The use of agnostic bigrams is penalized; they are weighted by a form of a 'prior'.
        ):
        self.zero_layers_over_wordlength = zero_layers_over_wordlength
        self.use_agnostic_bigram = use_agnostic_bigram
        self.occupation_threshold = occupation_threshold
        self.unibi_consistency_penalty = unibi_consistency_penalty
        self.confidence_unigrams = confidence_unigrams
        self.confidence_bigrams = confidence_bigrams
        self.agnostic_bigram_penalty = agnostic_bigram_penalty
        if use_all_unigram_combinations_as_bigrams and use_agnostic_bigram:
            raise ValueError('Using the agnostic bigram is pointless/incompatible with including all possible bigrams.')
        if unigram_levels != []:
            if possible_unigrams == []:
                self.possible_unigrams = [chr(i) for i in itertools.chain(range(ord('a'), ord('z') + 1), range(ord('0'), ord('9') + 1))]
            else:
                self.possible_unigrams = possible_unigrams
            self.unigram_levels = unigram_levels

        if bigram_levels != []:
            if possible_bigrams == [] and bigram_levels != []:
                if not use_all_unigram_combinations_as_bigrams:
                    self.possible_bigrams, _ = get_most_common_n_grams(use_agnostic_bigram=use_agnostic_bigram)
                else:
                    self.possible_bigrams, _ = get_most_common_n_grams(use_all_unigram_combinations=self.possible_unigrams)
            else:
                self.possible_bigrams = possible_bigrams
            self.bigram_levels = bigram_levels

    def chop_phoc(self, phoc_vector: np.ndarray):
        """
        Decomposes a PHOC vector in layer parts.
        Returns a list of tuples; each tuple is type (PhocLayerPart, np.ndarray)
        """
        Lu = len(self.possible_unigrams)
        Lb = len(self.possible_bigrams)
        phoc_uni_length = 0
        phoc_bi_length = 0
        current_begin = 0
        current_end = 0
        res = []
        for i in self.unigram_levels:
            for j in range(1, i+1):
                current_layerpart = PhocLayerPart(j, i, self.occupation_threshold, is_bigram=False)
                current_end = current_end + Lu
                phoc_uni_length = phoc_uni_length + Lu
                res.append( (current_layerpart, phoc_vector[current_begin:current_end]) )
                current_begin = current_end
        for i in self.bigram_levels:
            for j in range(1, i+1):
                current_layerpart = PhocLayerPart(j, i, self.occupation_threshold, is_bigram=True)
                current_end = current_end + Lb
                phoc_bi_length = phoc_bi_length + Lb
                res.append( (current_layerpart, phoc_vector[current_begin:current_end]) )
                current_begin = current_end
        if(phoc_uni_length + phoc_bi_length != len(phoc_vector)):
            raise ValueError('Output length={} should have been ={} (lvls: uni{}, bi{})'.format(phoc_uni_length + phoc_bi_length, len(phoc_vector), self.unigram_levels, self.bigram_levels))
        return res

    def get_phoc_layerpart(self, phoc_vector: np.ndarray, layerpart: PhocLayerPart, pdfnormalize = True):
        """
        :phoc_vector:       Input phoc, provided a single one-dimensional ndarray.
        :layerpart:         The particular part of the phoc that we want to extract.
        :pdfnormalize:      If true, the output vector will be normalized to be a pdf.
        """
        # Pick the weights that are relevant, if weighting by 'confidence' is required.
        if not layerpart.is_bigram:
            confidence_weights = self.confidence_unigrams
        else:
            confidence_weights = self.confidence_bigrams
        confidence_base = confidence_weights[0]
        confidence_decay = confidence_weights[1] * (layerpart.level-1) #So level 1 will have zero decay
        # ..and combine to get the right 'credibility' to use as the weight
        credibility = confidence_base * np.exp(-confidence_decay)
        for i in self.chop_phoc(phoc_vector):
            if(layerpart.compare(i[0])):
                res = i[1]**credibility + 1e-6
                if pdfnormalize:
                    res = res / np.sum(res)
                    if self.use_agnostic_bigram and layerpart.is_bigram:
                        res[-1] = res[-1] * self.agnostic_bigram_penalty
                    res = res / np.sum(res)
                return res
        raise ValueError('Unexpected error: Failed to get requested layerpart.')

    def layer_parts(self, layertype='all'):
        """

        :param layertype: Possible values: 'all', 'unigrams', 'bigrams'
        :return: Returns all PHOC layer parts (histograms) in a list. Each PHOC layer part is a PhocLayerPart object.
        """
        if layertype not in ['unigrams', 'bigrams', 'all']:
            raise ValueError
        res = []
        if layertype == 'all':
            res = self.layer_parts(layertype='unigrams') + self.layer_parts(layertype='bigrams')
            return res
        elif layertype == 'unigrams':
            for level in self.unigram_levels:
                for region in range(level):
                    res.append(PhocLayerPart(region+1, level, self.occupation_threshold, is_bigram=False))
            return res
        elif layertype == 'bigrams':
            for level in self.bigram_levels:
                for region in range(level):
                    res.append(PhocLayerPart(region+1, level, self.occupation_threshold, is_bigram=True))
            return res
        else:
            raise ValueError

    def phoc_size(self):
        phoc_unigrams = self.possible_unigrams
        unigram_levels = self.unigram_levels
        phoc_bigrams = self.possible_bigrams
        bigram_levels = self.bigram_levels

        ps = 0
        if phoc_unigrams is not None and unigram_levels is not None: #TODO: This if and the next one could go
            ps += len(phoc_unigrams) * np.sum(unigram_levels)
        if phoc_bigrams is not None and bigram_levels is not None:
            ps += len(phoc_bigrams) * np.sum(bigram_levels)
        return np.uint32(ps)

    def build_phoc_descriptor(self, words,
                              split_character=None, on_unknown_unigram='warn',
                              phoc_type='phoc'):
        '''
        Calculate Pyramidal Histogram of Characters (PHOC) descriptor (see Almazan 2014).

        Args:
            word (str): word to calculate descriptor for
            phoc_unigrams (str): string of all unigrams to use in the PHOC
            unigram_levels (list of int): the levels to use in the PHOC
            split_character (str): special character to split the word strings into characters
            on_unknown_unigram (str): What to do if a unigram appearing in a word
                is not among the supplied phoc_unigrams. Possible: 'warn', 'error', 'nothing'
            phoc_type (str): the type of the PHOC to be build. The default is the
                binary PHOC (standard version from Almazan 2014).
                Possible: phoc, spoc
        Returns:
            the PHOC for the given word
        '''
        # prepare output matrix
        phoc_unigrams = self.possible_unigrams
        unigram_levels = self.unigram_levels
        phoc_bigrams = self.possible_bigrams
        bigram_levels = self.bigram_levels

        phocs = np.zeros((len(words), self.phoc_size()))

        logger = logging.getLogger('PHOCGenerator')
        if on_unknown_unigram not in ['error', 'warn', 'nothing']:
            raise ValueError('I don\'t know the on_unknown_unigram parameter \'%s\'' % on_unknown_unigram)

        # map from character to alphabet position
        char_indices = {d: i for i, d in enumerate(phoc_unigrams)}

        # iterate through all the words
        for word_index, word in enumerate(words):
            if not isinstance(word, str):
                raise TypeError('Input word "{}" is not of class str (found class: {})'.format(word, type(word)) )
            if split_character is not None:
                word = word.split(split_character)
            n = len(word)  # pylint: disable=invalid-name
            for index, char in enumerate(word):
                if char not in char_indices:
                    if on_unknown_unigram == 'warn':
                        logger.warning('The unigram \'%s\' is unknown, skipping this character (full word was: %s)', char, word)
                        continue
                    elif on_unknown_unigram == 'error':
                        logger.fatal('The unigram \'%s\' is unknown', char)
                        raise ValueError()
                    else:
                        continue
                char_index = char_indices[char]
                for layerpart in self.layer_parts(layertype='unigrams'):
                    if layerpart.contains(index+1, n):
                        feat_vec_index = sum([l for l in unigram_levels if l < layerpart.level]) * len(phoc_unigrams) + (layerpart.region-1) * len(phoc_unigrams) + char_index
                        if phoc_type == 'phoc':
                            phocs[word_index, feat_vec_index] = 1
                        elif phoc_type == 'spoc':
                            phocs[word_index, feat_vec_index] += 1
                        else:
                            raise ValueError('The phoc_type \'%s\' is unknown' % phoc_type)
                        if self.zero_layers_over_wordlength and layerpart.level > n:
                            phocs[word_index, feat_vec_index] = 0
            # add bigrams
            if phoc_bigrams != []:
                ngram_features = np.zeros(len(phoc_bigrams) * np.sum(bigram_levels))
                for i in range(n - 1):
                    ngram = word[i:i + 2]
                    if ngram not in phoc_bigrams:
                        if self.use_agnostic_bigram:
                            ngram_index = phoc_bigrams['--']
                        else:
                            continue
                    else:
                        ngram_index = phoc_bigrams[ngram]
                    for layerpart in self.layer_parts(layertype='bigrams'):
                        if layerpart.contains(i+1, n, is_bigram=True):
                            feat_vec_index = sum([l for l in bigram_levels if l < layerpart.level]) * len(phoc_bigrams) + (layerpart.region - 1) * len(phoc_bigrams) + ngram_index
                            if phoc_type == 'phoc':
                                ngram_features[feat_vec_index] = 1
                            elif phoc_type == 'spoc':
                                ngram_features[feat_vec_index] += 1
                            else:
                                raise ValueError('The phoc_type \'%s\' is unknown' % phoc_type)
                            if self.zero_layers_over_wordlength and layerpart.level > n:
                                ngram_features[feat_vec_index] = 0
                phocs[word_index, -len(ngram_features):] = ngram_features
        return phocs

    def _convert_dict_to_list(self, d):
        res = [None] * len(d)
        for (k, v) in d.items():
            res[v] = k
        return res

    def analyze_phoc(self, phoc):
        if type(phoc) is not np.ndarray or len(phoc.shape) != 1:
            raise ValueError('Input to analyze_phoc should be a 1-dimensional numpy array (ie a single numpy vector).')
        unigrams = self.possible_unigrams
        bigrams = self.possible_bigrams
        unigram_levels = self.unigram_levels
        bigram_levels = self.bigram_levels

        i = 0
        print('Length of phoc: ' + str(len(phoc)))
        for ul in unigram_levels:
            for par in range(ul):
                for u in unigrams:
                    print('{:4}# {}/{} {:3} || {}'.format(i, par + 1, ul, u, phoc[i]))
                    i = i + 1

        for bl in bigram_levels:
            for par in range(bl):
                for b in self._convert_dict_to_list(bigrams):
                    print('{:4}# {}/{} {:3} || {}'.format(i, par + 1, bl, b, phoc[i]))
                    i = i + 1

    def compute_unigrambigram_cliquematrix(self, use_agnostic_bigram = False):
        """
        To be used as input to the LBP algorithm (lbp.py)
        :b0,b1,b2: See paper.
        :b3: Weight analogous to b2, but for the 'agnostic' bigram.
        :use_agnostic_bigram: See PhocLayout comments.
        :return:   A numpy matrix containing p(zeta_ij | g_i, g_j) = dirac(.) values (see paper)
        """
        b0 = self.unibi_consistency_penalty[0]
        b1 = self.unibi_consistency_penalty[1]
        b2 = self.unibi_consistency_penalty[2]
        b3 = self.unibi_consistency_penalty[3]
        if self.possible_unigrams == [] or self.possible_bigrams == []:
            #print('Warning: Will not compute unigrambigram_cliquematrix since both possible unigrams and bigrams arent available')
            return None
        res = np.ones((
            len(self.possible_bigrams),
            len(self.possible_unigrams),
            len(self.possible_unigrams),
        )) * (b0 + 1e-12)
        ulist = self.possible_unigrams
        blist = self._convert_dict_to_list(self.possible_bigrams)
        for (ib, b) in enumerate(blist):
            if use_agnostic_bigram and ib == len(blist)-1:
                if b != '--':
                    raise ValueError('Agnostic bigram is not denoted as "--".')
                for u0 in ulist:
                    for u1 in ulist:
                        ind0 = ulist.index(u0)
                        ind1 = ulist.index(u1)
                        if u0+u1 in blist:
                            continue
                        res[ib, ind0, ind1] = res[ib, ind0, ind1] + b3
            else:
                ind0 = ulist.index(b[0])
                ind1 = ulist.index(b[1])
                res[ib, ind0, :] = res[ib, ind0, :] + b1
                res[ib, :, ind1] = res[ib, :, ind1] + b1
                res[ib, ind0, ind1] = res[ib, ind0, ind1] + b2
        #Normalize
        for i in range(len(ulist)):
            for j in range(len(ulist)):
                res[:, i, j] = res[:, i, j] / np.sum(res[:, i, j])
        return res

    def get_ngram_idx(self, s):
        if len(s) == 1:
            a = self.possible_unigrams.index(s)
        elif len(s) == 2:
            if s in self.possible_bigrams:
                a = self.possible_bigrams[s]
            else:
                if not self.use_agnostic_bigram:
                    a = 0 #An alternative could have been: return a bigram with at least one shared letter
                else:
                    a = self.possible_bigrams['--']
        else:
            raise ValueError
        return a

class TestPhoc(unittest.TestCase):
    fixture_dummy = [
        'abc',
        'abbccc',
            ]
    fixture_english = [
        'dog',
        'devastatingly',
    ]
    fixture_bigramlist_dummy = {
        'ab':0, 'bc':1, 'ac':2
    }
    pl = PhocLayout(unigram_levels=[2, 3, 4], bigram_levels=[2])
    pl_dummy = PhocLayout(possible_unigrams='abc', unigram_levels=[1, 2])
    pl_dummy2 = PhocLayout(unigram_levels=[1])

    def test_english_unigrams_list(self):
        un = self.pl.possible_unigrams
        self.assertEqual(len(un), 36)
        self.assertEqual(un[0], 'a')
        self.assertEqual(un[-1], '9')

    def test_dummyphoc1(self):
        a = self.pl_dummy.build_phoc_descriptor(self.fixture_dummy)
        self.assertTrue(np.array_equal(a, [
            [1., 1., 1., 1., 1., 0., 0., 0., 1.],
            [1., 1., 1., 1., 1., 0., 0., 0., 1.],
        ]))

    def test_english_phoc(self):
        mpl = PhocLayout(unigram_levels=[1])
        a = mpl.build_phoc_descriptor(self.fixture_english)
        self.assertEqual(len(a), len(self.fixture_english))
        self.assertEqual(np.count_nonzero(a[0]), 3)  # 3 discrete letters in 'dog'
        self.assertEqual(np.count_nonzero(a[1]), 11) #11 discrete letters in 'devastatingly'

    def test_bigrams_sanity(self):
        self.assertEqual(len(self.pl.possible_bigrams), 75)

    def test_bigrams_dummylist(self):
        apl = PhocLayout(possible_unigrams='abc', unigram_levels=[1], possible_bigrams=self.fixture_bigramlist_dummy, bigram_levels=[2])
        a = apl.build_phoc_descriptor(self.fixture_dummy)
        self.assertTrue(np.array_equal(a, [
             [1., 1., 1., 1., 0., 0., 0., 1., 0.],
             [1., 1., 1., 1., 1., 0., 0., 0., 0.]
        ]))

    def test_analyzephoc(self):
        apl = PhocLayout(possible_unigrams='abc', unigram_levels=[1], possible_bigrams=self.fixture_bigramlist_dummy, bigram_levels=[2])
        a = apl.build_phoc_descriptor(self.fixture_dummy)
        for i, words in enumerate(self.fixture_dummy):
            #print('Description of "{}" as a PHOC vector, length {}'.format(words, len(a[i])))
            #apl.analyze_phoc(a[i])
            pass

        bpl = PhocLayout(unigram_levels=[1,2,4], bigram_levels=[2])
        a = bpl.build_phoc_descriptor(self.fixture_english)
        for i, words in enumerate(self.fixture_english):
            #print('Description of "{}" as a PHOC vector, length {}'.format(words, len(a[i])))
            #bpl.analyze_phoc(a[i])
            pass

    def test_againstlegacyphocs_dummy(self):
        uni_layers_set = [
            [1,2],
        ]
        bi_layers_set = [
            [1],
            [2],
        ]
        words = self.fixture_dummy
        for uni_layers in uni_layers_set:
            for bi_layers in bi_layers_set:
                legacy_result = legacy_phoc(words, possible_unigrams='abc', uni_layers=uni_layers, possible_bigrams=self.fixture_bigramlist_dummy, bi_layers=bi_layers)
                bpl = PhocLayout(possible_unigrams='abc', unigram_levels=uni_layers, possible_bigrams=self.fixture_bigramlist_dummy, bigram_levels=bi_layers)
                my_result = bpl.build_phoc_descriptor(words, on_unknown_unigram='nothing')
                self.assertTrue(np.array_equal(legacy_result, my_result))


    def test_againstlegacyphocs(self):
        uni_layers_set = [
            [],
            [1],
            [2],
            [3],
            [4],
            [2, 3, 4],
            [2, 3, 4, 5],
        ]
        bi_layers_set = [
            [2],
        ]
        words = load_strings('../fixtures/GW20/GW10_firsthalf.txt')
        for uni_layers in uni_layers_set:
            for bi_layers in bi_layers_set:
                legacy_result = legacy_phoc(words, uni_layers=uni_layers, bi_layers=bi_layers)
                bpl = PhocLayout(unigram_levels=uni_layers, bigram_levels=bi_layers)
                my_result = bpl.build_phoc_descriptor(words, on_unknown_unigram='nothing')
                self.assertTrue(np.array_equal(legacy_result, my_result))

    def test_multiplebigramlayers_1(self):
        bi_layers_set = [
            [1],
            [2],
        ]
        bi_layers_set.append(bi_layers_set[0] + bi_layers_set[1])
        words = [
            'abaabb',
            'aaaaaabbbbbb'
        ]
        possible_bigrams = {
            'ab': 0,
            'bb': 1,
        }
        my_result = [None] * len(bi_layers_set)
        for idx, bi_layers in enumerate(bi_layers_set):
            bpl = PhocLayout(possible_unigrams='ab', unigram_levels=[], possible_bigrams=possible_bigrams, bigram_levels=bi_layers)
            my_result[idx] = bpl.build_phoc_descriptor(words, on_unknown_unigram='nothing') #because that would trigger 'on_unknown_unigram'
            #my_result[idx] = legacy_phoc(words, possible_unigrams='ab', uni_layers=[], possible_bigrams=possible_bigrams, bi_layers=bi_layers) # This should fail anyway (Bug at sudholt's code)
        resA = my_result[-1]
        resB = np.concatenate((my_result[0], my_result[1]), axis=1)
        self.assertTrue(np.array_equal(resA, resB))

    def test_multiplebigramlayers_2(self):
        words = [
            'abaabb',
            'aaaaaabbbbbb'
        ]
        possible_bigrams = {
            'ab': 0,
            'bb': 1,
        }
        bpl = PhocLayout(possible_unigrams='ab', unigram_levels=[1,2], possible_bigrams=possible_bigrams, bigram_levels=[1,2])
        my_result = bpl.build_phoc_descriptor(words)
        req_result = [
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1.],
            [1., 1., 1., 0., 0., 1., 1., 1., 0., 0., 1., 1.], #Note that the 'ab' bigram counts as part of the second half, by convention
        ]
        self.assertTrue(np.array_equal(my_result, req_result))

    def test_multiplebigramlayers_allunigram_combinations_asbigrams(self):
        words = [
            'abaabb',
            'aaaaaabbbbbb'
        ]
        bpl = PhocLayout(possible_unigrams='ab', unigram_levels=[1,2], bigram_levels=[1,2], use_all_unigram_combinations_as_bigrams=True)
        my_result = bpl.build_phoc_descriptor(words)
        req_result = [
            # a    b    a12  b12  a22  b22  aa   ab   ba   bb   aa12 ab12 ba12 bb12 aa22 ab22 ba22 bb22
            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  1.,  0.,  1.],
            [ 1.,  1.,  1.,  0.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  1.],
        ]
        self.assertTrue(np.array_equal(my_result, req_result))

    def test_zero_toofinelayers(self):
        words = [
            'abab',
        ]
        bpl = PhocLayout(possible_unigrams='ab', unigram_levels=[1,2,3,4,5,6])
        my_result = bpl.build_phoc_descriptor(words)
        req_result_legacy = [
            #  1 |      2    |           3      |          4           |             5               |                 6                 |
            [1.,1.,1.,1.,1.,1.,1.,0.,1.,1.,0.,1.,1.,0.,0.,1.,1.,0.,0.,1.,1.,0.,0.,1.,0.,0.,1.,0.,0.,1.,1.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,1.],
        ]
        self.assertTrue(np.array_equal(my_result, req_result_legacy))
        bpl = PhocLayout(possible_unigrams='ab', unigram_levels=[1,2,3,4,5,6], zero_layers_over_wordlength=True)
        my_result = bpl.build_phoc_descriptor(words)
        req_result = [
            #  1 |      2    |           3      |          4           |             5               |                 6                 |
            [1.,1.,1.,1.,1.,1.,1.,0.,1.,1.,0.,1.,1.,0.,0.,1.,1.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        ]
        self.assertTrue(np.array_equal(my_result, req_result))

    def test_allunigram_combinations_asbigrams(self):
        word_length = 6
        pl = PhocLayout(unigram_levels=[1], bigram_levels=[1], use_all_unigram_combinations_as_bigrams=True)
        input_phoc = pl.build_phoc_descriptor(['astounding'])
        self.assertEqual( len(np.squeeze(input_phoc)), 1332 )
        self.assertEqual( len(np.squeeze(input_phoc)), pl.phoc_size() )

    def test_allunigram_combinations_asbigrams_2(self):
        word_length = 6
        pl = PhocLayout(unigram_levels=[1,2], bigram_levels=[1,2], use_all_unigram_combinations_as_bigrams=True)
        input_phoc = pl.build_phoc_descriptor(['astounding'])
        self.assertEqual( len(np.squeeze(input_phoc)), 3996 )
        self.assertEqual( len(np.squeeze(input_phoc)), pl.phoc_size() )

    def test_layerpart_containedletters_1(self):
        word_length = 6
        pl = PhocLayout(unigram_levels=[1,2], bigram_levels=[1])
        lparts = pl.layer_parts()
        #print(lparts)
        for j in lparts:
            pass
            #print(j)
            #print('Layer part {} contains ngrams {}'.format(j, j.contained_ngrams(word_length)))

    def test_layerpart_containedletters_2(self):
        word_length = 10
        pl = PhocLayout(unigram_levels=[1,2,3,12], bigram_levels=[1,2])
        lparts = pl.layer_parts()
        #print(lparts)
        for j in lparts:
            pass
            #print(j)
            #print('Layer part {} contains ngrams {}'.format(j, j.contained_ngrams(word_length)))

    def test_chop_phoc_and_reassemble(self):
        uni_layers_set = [
            #[], #this would cause a problem with 'on_unknown_unigram' -- not using unigrams is not a priority anyway
            [1],
            [2],
            [3],
            [4],
            [2, 3, 4],
            [2, 3, 4, 5],
        ]
        bi_layers_set = [
            [2],[2,4]
        ]
        words = load_strings('../fixtures/GW20/GW10_firsthalf.txt')
        for uni_layers in uni_layers_set:
            for bi_layers in bi_layers_set:
                bpl = PhocLayout(unigram_levels=uni_layers, bigram_levels=bi_layers)
                allphocs = bpl.build_phoc_descriptor(words)
                for idx, j in enumerate(range(allphocs.shape[0])):
                    phoc = allphocs[j, :]
                    chopped = bpl.chop_phoc(phoc)
                    res = []
                    for i in chopped:
                        ipart = i[1]
                        res = np.concatenate([res, ipart])
                    self.assertTrue(np.array_equal(phoc,res) )

    def test_buildphoc_accepts_only_ascii(self):
        words = np.array([b'270', b'and', b'instructions', b'october', b'publick', b'by', b'and', b'two', b'two', b'of'], dtype='|S15')
        pl = PhocLayout(unigram_levels=[1,2], bigram_levels=[1,2])
        with self.assertRaises(TypeError):
            allphocs = pl.build_phoc_descriptor(words)



if __name__ == '__main__':
    unittest.main()
