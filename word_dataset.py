import io,os
from typing import Type
import numpy as np
from skimage import io as img_io
import torch
from torch.utils.data import Dataset
from os.path import isfile
from skimage.transform import resize
from auxiliary_functions import image_resize, centered
from phoc.phoc import PhocLayout
import tqdm

class WordLineDataset(Dataset):
    #
    # TODO list:
    #
    #   Create method that will print data statistics (min/max pixel value, num of channels, etc.)   
    '''
    This class is a generic Dataset class meant to be used for word- and line- image datasets.
    It should not be used directly, but inherited by a dataset-specific class.
    '''
    def __init__(self, 
        basefolder: str = 'datasets/',                #Root folder
        subset: str = 'all',                          #Name of dataset subset to be loaded. (e.g. 'all', 'train', 'test', 'fold1', etc.)
        segmentation_level: str = 'line',             #Type of data to load ('line' or 'word')
        fixed_size: tuple =(128, None),               #Resize inputs to this size
        transforms: list = None,                      #List of augmentation transform functions to be applied on each input
        character_classes: list = None,               #If 'None', these will be autocomputed. Otherwise, a list of characters is expected.
        add_whitespace: bool = False,                  #Add leading and trailing whitespace in transcription
        filter_character_classes: bool = True,         #*Extra* reduce classes (besides what inheritors do on their own)
        ):
        self.basefolder = basefolder
        self.subset = subset
        self.segmentation_level = segmentation_level
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.setname = None                             # E.g. 'IAM'. This should coincide with the folder name
        self.stopwords = []
        self.stopwords_path = None
        self.character_classes = character_classes
        self.add_whitespace = add_whitespace
        self.filter_character_classes = filter_character_classes

    def __finalize__(self):
        '''
        Will call code after descendant class has specified 'key' variables
        and ran dataset-specific code
        '''
        assert(self.setname is not None)
        if self.stopwords_path is not None:
            for line in open(self.stopwords_path):
                self.stopwords.append(line.strip().split(','))
            self.stopwords = self.stopwords[0]
        save_file = 'saved_datasets/{}_{}_{}.pt'.format(self.subset, self.segmentation_level, self.setname) #dataset_path + '/' + set + '_' + level + '_IAM.pt'
        if isfile(save_file) is False:
            data = self.main_loader(self.subset, self.segmentation_level)
            #torch.save(data, save_file)   #Uncomment this in 'release' version
        else:
            data = torch.load(save_file)

        if self.filter_character_classes:
            filtered_data = []
            for d in tqdm.tqdm(data):
                filtered_data.append(
                    (d[0], self.filter_transcription(d[1]))
                )
            data = filtered_data
        if self.character_classes is None:
            res = set()
             #compute character classes given input transcriptions
            for _,transcr in tqdm.tqdm(data):
                res.update(list(transcr))
            res = sorted(list(res))
            print('Character classes: {} ({} different characters)'.format(res, len(res)))
            self.character_classes = res
        #########################################
        # Resize images
        if self.fixed_size is not None:
            print('**** word_dataset::__finalize__: Performing resizing to {}'.format(self.fixed_size))
            for idx, d in enumerate(tqdm.tqdm(data)):
                #img = image_resize(img, height=int(1.0 * nheight), width=int(1.0 * nwidth))
                #img = centered(img, (fheight, fwidth), border_value=0.0)
                word_img = d[0]
                tt = self.check_size(img=word_img, min_image_width_height=self.fixed_size[0])
                #word_img = resize(image=word_img, output_shape=).astype(np.float32)
                data[idx] = (
                    image_resize(tt, height=self.fixed_size[0], width=self.fixed_size[1]),
                    d[1]
                )
        ##########################################
        # Add transcriptions to data
        # Create PHOC layout target and setup network output size
        phoclayout = PhocLayout(
            #unigram_levels=[1],
            unigram_levels=[2,3,4,5], 
            bigram_levels=[2], 
            #use_all_unigram_combinations_as_bigrams=True,
            possible_unigrams = ''.join(self.character_classes),
            use_agnostic_bigram=False,
            )
        data = [(d[0], d[1], np.squeeze(phoclayout.build_phoc_descriptor([d[1]]))) for d in data]
        self.data = data
        #END FINALIZE

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index][0]
        ## Process transcription
        if self.add_whitespace:
            transcr = " " + self.data[index][1] + " "
        else:
            transcr = self.data[index][1]

        '''
        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]
        if self.subset == 'train':
            nwidth = int(np.random.uniform(.5, 1.5) * img.shape[1])
            nheight = int((np.random.uniform(.8, 1.2) * img.shape[0] / img.shape[1]) * nwidth)
        else:
            nheight, nwidth = img.shape[0], img.shape[1]

        try:
            nheight, nwidth = max(4, min(fheight-16, nheight)), max(8, min(fwidth-32, nwidth))
        except TypeError:
            print('warning -- word_dataset::__getitem__: Height or Width fixed size is set to None.')
        img = image_resize(img, height=int(1.0 * nheight), width=int(1.0 * nwidth))
        img = centered(img, (fheight, fwidth), border_value=0.0)
        '''
        if self.transforms is not None:
            for tr in self.transforms:
                if np.random.rand() < .5:
                    img = tr(img)
        # pad with zeroes
        #img = centered(img, (fheight, fwidth), np.random.uniform(.2, .8, size=2), border_value=0.0)
        img = torch.Tensor(img).float().unsqueeze(0)
        return img, transcr, self.data[index][2]

    def compute_queries(self):
        transcrs = [tr for _,tr,_ in self.data]
        uwords = np.unique(transcrs)
        udict = {w: i for i, w in enumerate(uwords)}
        lbls = np.asarray([udict[w] for w in transcrs])
        cnts = np.bincount(lbls)
        # From Almazan et al.,ICCV 2013: 
        # "We do not use stopwords or words that only appear once in the dataset as queries [in IAM]. 
        # However, those words are still present on the dataset and act as distractors."
        queries = [w for w in uwords if w not in self.stopwords and cnts[udict[w]] > 1 and len(w) > 1]
        for w in queries:
            print('Query {} exists {} times in given (test) set.'.format(w, cnts[udict[w]]))
        return(queries, lbls)
        
    def main_loader(self, subset, segmentation_level) -> list:
        # This function should be implemented by an inheriting class.
        raise NotImplementedError

    def check_size(self, img, min_image_width_height, fixed_image_size=None):
        '''
        checks if the image accords to the minimum and maximum size requirements
        or fixed image size and resizes if not
        
        :param img: the image to be checked
        :param min_image_width_height: the minimum image size
        :param fixed_image_size:
        '''
        if fixed_image_size is not None:
            if len(fixed_image_size) != 2:
                raise ValueError('The requested fixed image size is invalid!')
            new_img = resize(image=img, output_shape=fixed_image_size[::-1], mode='constant')
            new_img = new_img.astype(np.float32)
            return new_img
        elif np.amin(img.shape[:2]) < min_image_width_height:
            if np.amin(img.shape[:2]) == 0:
                print('OUCH: {}, {}'.format(img.shape, min_image_width_height))
                return None
            scale = float(min_image_width_height + 1) / float(np.amin(img.shape[:2]))
            new_shape = (int(scale * img.shape[0]), int(scale * img.shape[1]))
            new_img = resize(image=img, output_shape=new_shape, mode='constant')
            new_img = new_img.astype(np.float32)
            return new_img
        else:
            return img
    
    def print_random_sample(self, image, transcription, id, approx_num_of_samples=5000, as_saved_files=True):
        import random    #   Create method that will show example images using graphics-in-console (e.g. TerminalImageViewer)
        from PIL import Image
        # Run this with a very low probability
        x = random.randint(0, approx_num_of_samples)
        if(x > 20):
            return
        def show_image(img):
            def get_ansi_color_code(r, g, b):
                if r == g and g == b:
                    if r < 8:
                        return 16
                    if r > 248:
                        return 231
                    return round(((r - 8) / 247) * 24) + 232
                return 16 + (36 * round(r / 255 * 5)) + (6 * round(g / 255 * 5)) + round(b / 255 * 5)
            def get_color(r, g, b):
                return "\x1b[48;5;{}m \x1b[0m".format(int(get_ansi_color_code(r,g,b)))
            h = 12
            w = int((img.width / img.height) * h)
            img = img.resize((w,h), Image.ANTIALIAS)
            img_arr = np.asarray(img)
            h,w  = img_arr.shape #,c
            for x in range(h):
                for y in range(w):
                    pix = img_arr[x][y]
                    print(get_color(pix, pix, pix), sep='', end='')
                    #print(get_color(pix[0], pix[1], pix[2]), sep='', end='')
                print()
        if(as_saved_files):
            Image.fromarray(np.uint8(image*255.)).save('/tmp/a{}_{}.png'.format(id, transcription))
        else:
            print('Id = {}, Transcription = "{}"'.format(id, transcription))
            show_image(Image.fromarray(np.uint8(255.0*image)))
    
    def filter_transcription(self, input_transcription: str):
        # Inheriting classes will optionally use an elaborate mapping scheme.
        # E.g. map greek/arabic letters to latin ones.
        res = ''.join([c if (c.isalnum() or c=='_' or c==' ') else '*' for c in input_transcription.lower()])
        return(res)

class LineListIO(object):
    '''
    Helper class for reading/writing text files into lists.
    The elements of the list are the lines in the text file.
    '''
    @staticmethod
    def read_list(filepath, encoding='ascii'):        
        if not os.path.exists(filepath):
            raise ValueError('File for reading list does NOT exist: ' + filepath)
        
        linelist = []        
        if encoding == 'ascii':
            transform = lambda line: line.encode()
        else:
            transform = lambda line: line 

        with io.open(filepath, encoding=encoding) as stream:            
            for line in stream:
                line = transform(line.strip())
                if line != '':
                    linelist.append(line)                    
        return linelist

    @staticmethod
    def write_list(file_path, line_list, encoding='ascii', 
                   append=False, verbose=False):
        '''
        Writes a list into the given file object
        
        file_path: the file path that will be written to
        line_list: the list of strings that will be written
        '''                
        mode = 'w'
        if append:
            mode = 'a'
        
        with io.open(file_path, mode, encoding=encoding) as f:
            if verbose:
                line_list = tqdm.tqdm(line_list)
              
            for l in line_list:
                #f.write(unicode(l) + '\n')   Python 2
                f.write(l + '\n')
