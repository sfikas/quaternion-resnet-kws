import os
import re
import numpy as np 
from skimage import io as img_io
from word_dataset import WordLineDataset, LineListIO
import unicodedata as ud

latin_letters = {}

def is_latin(uchr):
    try: return latin_letters[uchr]
    except KeyError:
         return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))

def only_roman_chars(unistr):
    return all(is_latin(uchr) for uchr in unistr if uchr.isalpha()) # isalpha suggested by John Machin

def only_nonroman_chars(unistr):
    return all(not is_latin(uchr) for uchr in unistr if uchr.isalpha()) # isalpha suggested by John Machin

class PiopDataset(WordLineDataset):
    def __init__(self, basefolder, subset, segmentation_level, fixed_size, transforms, character_classes):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, transforms, character_classes)
        self.setname = 'piop'
        self.root_dir = 'datasets/CULDILE-kws'
        if basefolder is not None:
            self.root_dir = basefolder
        super().__finalize__()

    def main_loader(self, partition, level) -> list:
        if partition not in [None, 'train', 'test', 'trainval']:
            raise ValueError('partition must be one of None, train, trainval or test')
        self.word_list = None
        self.word_string_embeddings = None
        self.query_list = None
        self.label_encoder = None
        data = []
        idx = 0
        doc_idx = 0
        cutoff_pct = 0.8
        if(level == 'word'):
            json_filenames = sorted([elem for elem in os.listdir(self.root_dir) if elem.endswith('.json')])
            total_docs = len(json_filenames)
            for json_filename in json_filenames:
                doc_idx += 1
                if(partition == 'train' and doc_idx+1 > np.round( cutoff_pct * total_docs) ): #TODO: This has to change to sth more generic
                    continue
                elif(partition == 'test' and doc_idx+1 <= np.round( cutoff_pct * total_docs) ):
                    continue
                else:
                    pass
                current_wordlist = self.getTextSegments(os.path.join(self.root_dir, json_filename))
                img_filename = '.'.join([os.path.splitext(json_filename)[0], 'jpg'])
                doc_img = img_io.imread(os.path.join(self.root_dir, img_filename), plugin='pil')
                doc_img = 1 - doc_img.astype(np.float32) / 255.0
                doc_img = np.mean(doc_img, axis=-1)
                for (text, coords) in current_wordlist:
                    #TODO 1: Many words span two lines, with a non-convex polygon surrounding them. Don't load them
                    #TODO 2: Map all characters to lower-case, non-accented
                    #TODO 3: Some 'words' actually span two words: px "Μυκάλης 23"
                    text = text.replace(' ', '') #Strip whitespace in-between as well (this shouldnt exist in the gt in the first place, cf TODO3, but..)
                    if not only_nonroman_chars(text):
                        continue
                    if(text == ''):
                        continue
                    coords = np.array(coords)
                    top = np.min(coords[:, 1])
                    bottom = np.max(coords[:, 1])
                    left = np.min(coords[:, 0])
                    right = np.max(coords[:, 0])
                    token_img = doc_img[int(top):int(bottom), int(left):int(right)].copy()
                    token_img = self.check_size(img=token_img, min_image_width_height=self.fixed_size[0])
                    if(token_img is None):
                        continue
                    #text = line['TextEquiv']['Unicode']
                    data.append(
                        (token_img, text)
                    )
                    idx += 1
                    if idx % 1000 == 0:
                        print('imgs: [{}/]'.format(idx))
                    self.print_random_sample(token_img, text, idx, approx_num_of_samples=4860, as_saved_files=False)
        elif(level == 'line'):
            raise NotImplementedError
        else:
            raise ValueError
        return(data)

    def getTextSegments(self, input_json):
        """
        getTextSegments
            Returns a list of all segments in a "Culdile-type" json.
            Each segment is of a specific semantic class.

        :param input_json: A culdile json.
        :return: List of segmets. Each segment object contains a list of coordinates of the surrounding polygon, *and* a class identifier (a natural number).
        """
        def parseCoordinateList(l):
            res = []
            #class_number = int(l[0])
            class_name = l[0]
            polygon_data = l[1]

            coords_regex = r'\[\s*([\d|\.|\s]*),([\d|\.|\s]*)\s*\]'
            coordpairs = re.findall(coords_regex, polygon_data)
            for i in coordpairs:
                res.append([int(float(i[0])),int(float(i[1]))])
            return(
                (class_name,res)
            )
        #textline_regex = r'<TextLine id=[.\s\S]*?<Coords points="(?P<textline>[.\s\S]*?)"/>[.\s\S]*?</TextLine>'
        textline_regex = r'"label": "([\w|\s]*)",\s* "points":\s*([\[|\]|\d|\.|\,|\s]*)|\s*\],\s*"group_id"'
        try:
            textfile = open(input_json, 'r')
            matches = re.findall(textline_regex, textfile.read())
            textfile.close()
        except:
            print('IO Error while trying to read text segements from {}.Skipping..'.format(input_json))
            return(None)
        return([parseCoordinateList(m) for m in matches])
