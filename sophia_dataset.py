import os
import re
import cv2
import numpy as np 
from skimage import io as img_io
from word_dataset import WordLineDataset
from os.path import isfile
from auxiliary_functions import image_resize, centered
from skimage.transform import resize
#
import xmltodict

class SophiaDataset(WordLineDataset):
    def __init__(self, basefolder, subset, segmentation_level, fixed_size, transforms, character_classes):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, transforms, character_classes)
        self.setname = 'sophia'
        self.root_dir = 'datasets/{}/data'.format(self.setname)
        self.data = [] # A list of tuples (image data, transcription)
        self.query_list = None
        self.datafilelist = []
        for i in range(1, 48):
            if i == 12:
                continue
            originalPageFile = '_{:04d}.JPG'.format(i)
            binarizedPageFile = '_{:04d}_b.tif'.format(i)
            xmlFile = '_{:04d}.xml'.format(i)
            self.datafilelist.append(
                (originalPageFile, binarizedPageFile, xmlFile)
            )
        super().__finalize__()

    def compute_queries(self):
        transcrs = [tr for _,tr,_ in self.data]
        uwords = np.unique(transcrs)
        udict = {w: i for i, w in enumerate(uwords)}
        lbls = np.asarray([udict[w] for w in transcrs])
        cnts = np.bincount(lbls)
        # From Sfikas et al. 2015:
        #For the handwritten Memoirs set we choose all words that have
        #more than 5 letters and 4 instances as queries, for a total of 21 queries.
        # Note: It is "21 queries" because in the original publication (Gatos et al.) the methods used the whole set for tests
        # (the baselines were learning-free!)
        #queries = [w for w in uwords if w not in self.stopwords and cnts[udict[w]] > 4 and len(w) > 5]
        #queries = ['γνωστὸν', 'θυγάτηρ', 'μεταξὺ', 'μητέρα', 'μητρός',
        #        'μᾶλλον', 'οἰκίαν', 'οἰκίας', 'οἰκογενείας', 'οὐδέποτε', 'πατέρα',
        #        'πατρός', 'πλησίον', 'πρεσβείαν', 'πρεσβείας', 'τέλους', 'ταύτην', 'τοσοῦτον', 'ἐγένετο', 'ἕνεκεν', 'ἡμέραν']
        # Update: This is problematic as well: Some queries only have one instance or none at all in test        
        queries = ['μητέρα', 'μητρός', 'μᾶλλον', 'οἰκογενείας', 'οὐδέποτε', 'πατέρα',
                'πατρός', 'πλησίον', 'πρεσβείαν', 'πρεσβείας', 'ταύτην', 'τοσοῦτον', 'ἕνεκεν', 'ἡμέραν']
        for w in queries:
            print('Query {} exists {} times in given (test) set.'.format(w, cnts[udict[w]]))
        return(queries, lbls)

    def main_loader(self, partition, level) -> list:
        ##########################################
        # Load pairs of (image, ground truth)
        ##########################################
        # load the dataset
        data = []
        if(level == 'word'):
            word_id = 1
            datafilelist = self.datafilelist
            for (originalPageFile, binarizedPageFile, xmlFile) in datafilelist:
                doc_img = img_io.imread(os.path.join(self.root_dir, originalPageFile), plugin='pil')
                doc_img = 1 - doc_img.astype(np.float32) / 255.0
                for word in self.get_words_from_pagexml(os.path.join(self.root_dir, xmlFile)):
                    x, y, w, h = word[1]
                    word_img = doc_img[y:y+h, x:x+w].copy()
                    #tt = doc_img[y:y+h, x:x+w, :].copy()
                    #self.print_random_sample(tt, '', 0, approx_num_of_samples=100, as_saved_files=True)
                    word_img = self.check_size(img=word_img, min_image_width_height=self.fixed_size[0])
                    # Decide on split_id (this comes from footnote on page 3 of Sfikas et al.2015)
                    if(len(word_img.shape) == 3):
                        word_img = np.mean(word_img, axis=-1)
                    if word_id >= 1 and word_id <= 2000:
                        current_split_id = 'train'
                    elif word_id >= 2001 and word_id <= 4000:
                        current_split_id = 'test'
                    elif word_id >= 4001 and word_id <= 4941:
                        current_split_id = 'val'
                    else:
                        raise ValueError('Word id read out of bounds (={}); it should have been in [1,4941].'.format(current_split_id))
                    word_id += 1                        
                    if current_split_id != partition:
                        continue
                    transcr = word[2]
                    data.append(
                        (word_img, transcr)
                    )
                    if word_id % 1000 == 0:
                        print('imgs: [{}/]'.format(word_id))
                    self.print_random_sample(word_img, transcr, word_id, approx_num_of_samples=4941, as_saved_files=False)
        elif(level == 'line'):
            if partition == 'train':
                datafilelist = self.datafilelist[0:25]
            elif partition == 'test':
                datafilelist = self.datafilelist[25:37]
            elif partition == 'val':
                datafilelist = self.datafilelist[37:47]
            elif partition is None:
                datafilelist = self.datafilelist
            else:
                raise ValueError('Invalid partition name, valid names are train, test, val.')
            lines_parsed = 0
            for (_, binarizedPageFile, xmlFile) in datafilelist:
                doc_img = img_io.imread(os.path.join(self.root_dir, binarizedPageFile), plugin='pil')
                doc_img = 1 - doc_img.astype(np.float32) / 255.0
                with open(os.path.join(self.root_dir, xmlFile), 'r') as f:
                    xmldata = f.read()
                    xmldoc = xmltodict.parse(xmldata)
                textlines = xmldoc['PcGts']['Page']['TextRegion']['TextLine']
                for line in textlines:
                    lines_parsed += 1
                    raw_coords = line['Coords']['@points']
                    coord_list = raw_coords.split(' ')
                    ys = []
                    xs = []
                    for coord in coord_list:
                        tt = coord.split(',')
                        ys.append(int(tt[0]))
                        xs.append(int(tt[1]))
                    top = np.min(xs)
                    bottom = np.max(xs)
                    left = np.min(ys)
                    right = np.max(ys)
                    text = line['TextEquiv']['Unicode']
                    token_img = doc_img[int(top):int(bottom), int(left):int(right)].copy()
                    token_img = self.check_size(img=token_img, min_image_width_height=self.fixed_size[0])
                    data.append(
                        (token_img, text)
                    )
                    self.print_random_sample(token_img, text, lines_parsed, approx_num_of_samples=385, as_saved_files=False)
            print('For partition {}, {} {} tokens have been parsed'.format(partition, lines_parsed, level))
        else:
            raise ValueError('Segmentation level must be either line or word.')
        return(data)


    def get_words_from_pagexml(self, xmlname,
        keep_punctuation=False,
        keep_capitals=False,
        keep_latins=False):
        """
        Returns a list of tuples. Each tuple corresponds to one word.
        """
        with open(xmlname, 'r') as f:
            xmldata = f.read().replace('\n', '')
        rexp = '<Word id="(.*?)">\s*<Coords points="(.*?)"/>\s*<TextEquiv>\s*<Unicode>(.+?)</Unicode>\s*</TextEquiv>\s*</Word>'
        words = re.findall(rexp, xmldata)
        #
        words_processed = []
        punctuation_mark_table = dict.fromkeys(map(ord, '\'‘&,.’:;"-()!·'), None)
        latin_min_mark_table = dict.fromkeys(map(ord, 'abcdefghijklmnopqrstuvwxyz'), None)
        latin_maj_mark_table = dict.fromkeys(map(ord, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'), None)
        for x in words:
            if keep_punctuation is False:
                transcr_new = x[2].translate(punctuation_mark_table)
            else:
                transcr_new = x[2]
            transcr_new = transcr_new.replace('&quot', '')
            transcr_new = transcr_new.replace('quot', '')
            if keep_capitals is False:
                transcr_new = transcr_new.lower()
            if keep_latins is False:
                transcr_new = transcr_new.translate(latin_min_mark_table)
                transcr_new = transcr_new.translate(latin_maj_mark_table)
            #trascr_new = re.sub(r'&quot', '', transcr_new)
            points_new = self.process_bbox(x[1])
            #id_new = x[0]
            id_new = xmlname
            if(len(transcr_new) == 0):
                print('Warning! Found word with empty transcription (probably due to removed punctuation). Replacing with dummy character "#"')
                transcr_new = '#'
            words_processed.append( (id_new, points_new, transcr_new) )
        print('Found {} words in file {}.'.format(len(words_processed), xmlname))
        return(words_processed)

    def get_list_of_unigrams(self, corpus_file):
        charset = set()
        with open(corpus_file, 'r') as f:
            for word in f.readlines():
                word = word.strip()
                for char in word:
                    charset.add(char)
        print(charset)
        return(charset)

    def process_bbox(self, xx):
        xx = xx.split(' ')
        res = []
        for i in xx:
            tt = i.split(',')
            rj = []
            for j in tt:
                rj.append(int(j))
            res.append(rj)
        res = np.array(res)
        res = cv2.boundingRect(res)
        return(res)
