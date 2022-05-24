

class PhocLayerPart():
    def __init__(self, region, level,
                        occupation_threshold,
                        is_bigram=False,
                        ):
        if region < 0 or level < 0 or region > level:
            raise ValueError
        self.region = region
        self.level = level
        self.is_bigram = is_bigram
        self.occupation_threshold = occupation_threshold

    def compare(self, other):
        """
        Checks if self == other
        """
        if self.region == other.region and self.level == other.level and self.is_bigram == other.is_bigram:
            return True
        else:
            return False

    def contains(self, ngram_position, word_length, is_bigram=False):
        def occupancy(k, n, l):
            return [float(k) / n, float(k + l) / n]
        def overlap(a, b):
            return [max(a[0], b[0]), min(a[1], b[1])]
        def size(r):
            return r[1] - r[0]
        """
        Check whether a letter or bigram (ngram) is contained in this layer part.
        :param ngram_position: Position of the first letter of this letter/bigram/ngram (in [1,word_length])
        :param word_length: Length of the word, K
        :param is_bigram: Is this ngram a bigram or not.
        :return:
        """
        if ngram_position == 0:
            raise ValueError("ngram_position argument should start at position 1.")
        if ngram_position > word_length:
            raise ValueError("ngram_position cannot be greater than word_length")
        if is_bigram is False:
            ngram_length = 1
        else:
            ngram_length = 2
        region_occ = occupancy(self.region - 1, self.level, 1)
        ngram_occ = occupancy(ngram_position - 1, word_length, ngram_length)
        #Note the condition >=0.5, which follows Sudholt's implementation; from this follows that an ngram may belong to two adjacent layers.
        if size(overlap(ngram_occ, region_occ)) / size(ngram_occ) >= self.occupation_threshold:
            return True
        else:
            return False

    def contained_ngrams(self, word_length):
        """
        Returns a tuple with positions of all related letters for this layer part.
        """
        res = []
        if self.is_bigram:
            K = word_length - 1
        else:
            K = word_length
        for i in range(1, K+1):
            if self.contains(i, word_length, is_bigram=self.is_bigram):
                if not self.is_bigram:
                    res.append(i)
                else:
                    res.append( (i, i+1) )
        return tuple(res)

    def __str__(self):
        if self.is_bigram:
            return('Bigram layerpart, {}/{}'.format(self.region, self.level))
        else:
            return('Unigram layerpart, {}/{}'.format(self.region, self.level))
