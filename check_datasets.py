import argparse
import logging
from auxiliary_functions import affine_transformation
from sophia_dataset import SophiaDataset
from piop_dataset import PiopDataset

logger = logging.getLogger('Check_datasets')
logger.info('--- Check word-line datasets toot ---')
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# argument parsing
parser = argparse.ArgumentParser()
# - train arguments
parser.add_argument('--dataset', choices=['GW', 'IAM','RIMES','SOPHIA','Botany','PIOP'], default='IAM',
                    help='Which dataset to use. Default: IAM')                    
parser.add_argument('--segmentation_level', choices=['word', 'line'], default='line',
                    help='Segmentation level to use. Default: line')
#parser.add_argument('--filter_character_classes', action='store_true', help='Reduce character classes (usually uppercase to lowercase)')
parser.add_argument('--dataset_folder', required=False, #type=argparse.FileType('rb'),
                    default='datasets',
                    help='Root folder containing datasets.')
#parser.add_argument('--autocompute_character_classes', dest='autocompute_character_classes', 
#                    action='store_true', 
#                    help='Compute character classes provided the training set. These will be stored as a dataset property and printed on terminal.')  
#parser.set_defaults(#autocompute_character_classes=False
    #filter_character_classes = False,
#)
args = parser.parse_args()

logger.info('###########################################')
logger.info('######## Experiment Parameters ############')
for key, value in vars(args).items():
    logger.info('%s: %s', str(key), str(value))
logger.info('###########################################')

#aug_transforms =[lambda x: affine_transformation(x, s=.2)]
aug_transforms = None

if args.dataset == 'SOPHIA':
    myDataset = SophiaDataset
elif args.dataset == 'PIOP':
    myDataset = PiopDataset
else:
    raise NotImplementedError

fixed_size = (128, 1024)
train_set = myDataset(args.dataset_folder, 'train', args.segmentation_level, fixed_size=fixed_size, transforms=aug_transforms,
                        character_classes=None) #(128, 1024))
test_set = myDataset(args.dataset_folder, 'test', args.segmentation_level, fixed_size=fixed_size, transforms=None,
#                        character_classes=train_set.character_classes) #(None,None)
                         character_classes=None) #(128, 1024))
print(test_set.compute_queries())
#
#train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=8, drop_last=True)
#test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=8)