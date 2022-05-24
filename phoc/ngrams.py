def get_unigrams_from_strings(word_strings, split_character=None):
    if split_character is not None:
        unigrams = [elem for word_string in word_strings for elem in word_string.split(split_character)]
    else:
        unigrams = [elem for word_string in word_strings for elem in word_string]
    unigrams = sorted(set(unigrams))
    return unigrams

def get_most_common_n_grams(words=None, num_results=50, len_ngram=2, 
    use_all_unigram_combinations=None,
    use_agnostic_bigram=False,
    ):
    '''
    Calculates the 50 (default) most common bigrams (default) from a
    list of pages, where each page is a list of WordData objects.

    (GS) If no 'words' are provided, then the 75 most common bigrams are returned.
    These are the same as found in Almazan et al's code.

    @param words: (list of str)
        List containing the words from which to extract the bigrams
    @param num_results: (int)
        Number of n-grams returned.
    @param len_ngram: (int)
        length of n-grams.
    @return most common <n>-grams
    '''
    if words == None:
        if use_all_unigram_combinations is None:
            bigrams = [
            'er','in','es','ti','te','at','on','an','en','st',
            'al','re','is','ed','le','ra','ri','li','ar','ng',
            'ne','ic','or','nt','ss','ro','la','se','de','co',
            'ca','ta','io','it','si','us','ea','ac','el','ma',
            'na','ni','tr','ch','di','ia','et','to','un','ns',
            'll','ec','me','lo','sc','ol','as','he','ly','ce',
            'nd','il','pe','sa','mi','rs','ve','ou','th','sp',
            'ur','om','ha','sh','nc',
            ]
            if use_agnostic_bigram:
                bigrams.append('--')
        else:
            bigrams = []
            for i in use_all_unigram_combinations:
                for j in use_all_unigram_combinations:
                    bigrams.append(i + j)
        return ({k: i for i, k in enumerate(bigrams)}, bigrams)

    ngrams = {}
    for word in words:
        w_ngrams = get_n_grams(word, len_ngram)
        for ngram in w_ngrams:
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
    sorted_list = sorted(ngrams.items(), key=lambda x: x[1], reverse=True)
    top_ngrams = sorted_list[:num_results]
    return {k: i for i, (k, _) in enumerate(top_ngrams)}

def get_n_grams(word, len_ngram):
    '''
    Calculates list of ngrams for a given word.

    @param word: (str)
        Word to calculate ngrams for.
    @param len_ngram: (int)
        Maximal ngram size: n=3 extracts 1-, 2- and 3-grams.
    @return:  List of ngrams as strings.
    '''
    return [word[i:i + len_ngram]for i in range(len(word) - len_ngram + 1)]
