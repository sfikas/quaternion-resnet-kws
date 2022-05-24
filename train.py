import argparse
import logging

import tqdm
import numpy as np
import torch.cuda
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from datetime import datetime
from uuid import uuid4

cudnn.benchmark = True
import torch.nn.functional as F
from torchsummary import summary

from sophia_dataset import SophiaDataset
from piop_dataset import PiopDataset
import auxiliary_functions as utils
from models import ResnetCNN_v2

def augment_from_1_to_4_channels(x, device):
    N, _, H, W = x.shape
    paddingchannel = torch.zeros(N, 3, H, W).to(device)
    x = torch.cat([x, paddingchannel], 1)        
    return(x)

def unique_id():
    """
    unique_id
        This is used to create a unique string to be used as a folder name and model instances.
    """
    return(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-') + uuid4().hex[0:5])

logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('Hypercomplex Keyword Spotting::train')
logger.info('--- Running Hypercomplex Keyword Spotting Training ---')
# argument parsing
parser = argparse.ArgumentParser()
# - train arguments
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, #In install_info georgeretsi writes lr=1e-3 though
                    help='lr')
parser.add_argument('--dataset', choices=['SOPHIA', 'PIOP'], default='SOPHIA',
                    help='Which dataset to use. Default: SOPHIA')
parser.add_argument('--model', choices=['standard', 'resnet'], default='resnet')
parser.add_argument('--quaternion', dest='quaternion', action='store_true', help='Use the quaternionized model.')
parser.add_argument('--small_version', dest='small_version', action='store_true', help='Use a cut-down version of the model.')
parser.add_argument('--display', action='store', type=int, default=100,
                    help='The number of iterations after which to display the loss values. Default: 100')
parser.add_argument('--gpu_id', '-gpu', action='store',
                    type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                    default='0',
                    help='The ID of the GPU to use. Default: GPU 0.')
parser.add_argument('--batch_size', '-bs', action='store', type=int, default=80,
                    help='The batch size after which the gradient is computed.')
parser.add_argument('--max_epochs', '-me', action='store', type=int, default=900)
parser.add_argument('--fixed_size', '-fim', action='store',
                    type=lambda str_tuple: tuple([int(elem) for elem in str_tuple.split(',')]),
                    help='Specifies the images to be resized to a fixed size when presented to the CNN. Argument must be two comma seperated numbers.')
parser.add_argument('--network_cfg', '-cfg', action='store',
                    type=lambda str_list: utils.parse_cfg(str_list),
                    help="Specifies network backbone configuration. \
                            Example: 2,64XMX4,128 gives a (2,64) conv layer followed by max-pooling followed by a (4,128) conv layer.")
parser.add_argument('--dataset_folder', required=False,
                    default=None,
                    help='Folder containing datasets.')
parser.add_argument('--dryrun', dest='dryrun', action='store_true', help='Debug mode: Just load the datasets and quit before NN model creation.')
parser.add_argument('--print_network', dest='print_network', action='store_true', help='Just print network info.') #TODO implement
parser.set_defaults(dryrun=False,
    quaternion=False,
    small_version=False,
    print_network=False,
    #fixed_size=(128, None),
    fixed_size = (32, 128),
    #network_cfg='2,64 x M x 64,128 x M x 4,256' #cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)] #, (2, 512)]
    #network_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)], #, (2, 512)]
    network_cfg = [(2, 64), 'M'], #, (2, 512)]
)
args = parser.parse_args()
args.restart_epochs = args.max_epochs // 3
if(args.fixed_size[0] is None or args.fixed_size[1] is None):
    raise NotImplementedError("Don't use None for the fixed size arguments -- batch size should then be set to 1. (=bad bad bad)")

logger.info('###########################################')
logger.info('Experiment Parameters:')
for key, value in vars(args).items():
    logger.info('%s: %s', str(key), str(value))
logger.info('###########################################')
if not torch.cuda.is_available() or args.gpu_id[0] == -1:
    args.gpu_id = None
    device = torch.device('cpu')
    logger.warning('Could not find CUDA environment, using CPU mode (device: {})'.format(device))
else:
    device = torch.device('cuda:{}'.format(args.gpu_id[0]))
    logger.warning('Using GPU mode. Specifically our device is {}'.format(device))

np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})


logger.info('Loading dataset.')

if args.dataset == 'SOPHIA':
    myDataset = SophiaDataset
elif args.dataset == 'PIOP':
    myDataset = PiopDataset
else:
    raise NotImplementedError

train_set = myDataset(args.dataset_folder, 'train', 'word', fixed_size=args.fixed_size, 
            transforms=[lambda x: utils.affine_transformation(x, s=.2)], character_classes=None) #(128, 1024))
            #transforms=None, character_classes=None) #(128, 1024))
test_set = myDataset(args.dataset_folder, 'test', 'word', fixed_size=args.fixed_size, 
            transforms=None, character_classes=train_set.character_classes) #(None,None))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
logger.info('Preparing Net...')
if args.dryrun:
    exit(1)

phoc_size = len(test_set[0][2])
logger.info('Target size will be a descriptor of size {}.'.format(phoc_size))
if args.model == 'standard':
    raise NotImplementedError('Deprecated.')
elif args.model == 'resnet':
    CNN = ResnetCNN_v2
else:
    raise ValueError('Unknown model name.')
cnn = CNN(n_out=phoc_size, cnn_cfg=args.network_cfg, quaternionized=args.quaternion, small_version=args.small_version)
#cnn.init_weights()
cnn = cnn.to(device)
if args.gpu_id is not None and len(args.gpu_id) > 1:
    raise NotImplementedError('Parallelism across GPUs not implemented yet.')
    # nn.parallel.DistributedDataParallel
    cnn = nn.DataParallel(cnn, device_ids=args.gpu_id) 
loss = BCEWithLogitsLoss(size_average=False) #optionally use some weighting there, to accomodate for rare PHOC bins

if(args.print_network):
    logger.info('Printing info for a hypothetical {}x{} input to the network.'.format(args.fixed_size[0], args.fixed_size[1]))
    summary(cnn, (4, args.fixed_size[0], args.fixed_size[1]))
    total_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    print('CNN total trainable params = {}'.format(total_params))
    exit(0)

optimizer = torch.optim.Adam(cnn.parameters(), args.learning_rate, weight_decay=0.00005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.restart_epochs)

executionid = 'qkws-{}'.format(
    unique_id(),
)
logger.info('Using execution id: {} (search the subfolder of the same name in the results repertoire)'.format(
    executionid,
))
queries, word_numerical_labels = test_set.compute_queries()
if not queries:
    raise ValueError('No query words could be determined! (is the test set too small?)')
## Variables to save to npz
#
test_loss_history = []
train_loss_history = []
map_history = []
for epoch in range(1, args.max_epochs+1):
    cnn.train()
    losses = []
    for idx, (img, transcr, phoc) in enumerate(tqdm.tqdm(train_loader)):
        optimizer.zero_grad()
        img = img.to(device)
        phoc_predicted_logits = cnn(augment_from_1_to_4_channels(img, device))
        phoc_targets = phoc.to(device)
        #Image.fromarray(np.uint8(img[0].cpu().detach().numpy().squeeze() * 255.)).save('/tmp/a{}_{}.png'.format(idx,transcr[0]))
        loss_val = loss(phoc_predicted_logits, phoc_targets)
        loss_val.backward()
        optimizer.step()
        losses.append(loss_val.cpu().detach().numpy())
    train_loss_history.append(np.mean(np.array(losses)))
    logger.info('Epoch: {:03d} -----Binary CE loss {:.3f}'.format(epoch, train_loss_history[-1]))

    scheduler.step()
    if epoch % 2 == 0:
        # mode -> 0: QbE, 1:QbS, 2:both
        #distance = 'cosine'
        distance = 'euclidean'
        cnn.eval()
        logger.info('Testing KWS at epoch %d', epoch)
        tdecs = []
        transcrs = []
        loss_test = []
        for img, transcr, phoc in test_loader:
            img = img.to(device)
            with torch.no_grad():
                test_phoc_predicted_logits = cnn(augment_from_1_to_4_channels(img, device))
                test_predicted_phoc = torch.sigmoid(test_phoc_predicted_logits)
                if distance == 'euclidean':
                    tdec = test_predicted_phoc.cpu().numpy().squeeze()
                elif distance == 'cosine':
                    tdec = F.normalize(test_predicted_phoc, dim=1).cpu().numpy().squeeze()
                tdecs += [tdec]
                transcrs += [list(transcr)]
                tt = loss(test_phoc_predicted_logits, phoc.to(device))
                loss_test.append(tt.cpu().detach().numpy())
        test_loss_history.append(np.mean(loss_test))
        print('Loss_test: {}'.format(test_loss_history[-1]))
        tdecs = np.concatenate(tdecs)
        transcrs = np.concatenate(transcrs)

        qids = np.asarray([i for i,t in enumerate(transcrs) if t in queries])
        qdecs = tdecs[qids]
        if distance == 'euclidean':
            D = -2 * np.dot(qdecs, np.transpose(tdecs)) + \
                np.linalg.norm(tdecs, axis=1).reshape((1, -1))**2 + \
                np.linalg.norm(qdecs, axis=1).reshape((-1, 1))**2
        elif distance == 'cosine':
            D = -np.dot(qdecs, np.transpose(tdecs))
        # bce similarity
        #S = np.dot(qphocs_est, np.log(np.transpose(phocs_est))) + np.dot(1-qphocs_est, np.log(np.transpose(1-phocs_est)))
        Id = np.argsort(D, axis=1)
        while Id.max() > Id.shape[1]:
            Id = np.argsort(D, axis=1)
        ap = [utils.average_precision(word_numerical_labels[Id[i]][1:] == word_numerical_labels[qc]) for i, qc in enumerate(qids)]
        map_qbe = 100 * np.mean(ap)
        logger.info('QBE MAP at epoch %d: %f', epoch, map_qbe)
        map_history.append(map_qbe)
        cnn.train()
        ## END OF test_kws

    if epoch % 10 == 0:
        logger.info('Saving net after %d epochs', epoch)
        save_fn = 'results/qkws-cnn-{}.pt'.format(executionid)
        torch.save(cnn.cpu().state_dict(), save_fn)
        cnn = cnn.to(device)
        np.savez_compressed('results/qkws-cnn-{}.npz'.format(executionid),
            test_loss_history=test_loss_history,
            train_loss_history=train_loss_history,
            map_history=map_history,
        )

    if epoch % args.restart_epochs == 0:
        optimizer = torch.optim.Adam(cnn.parameters(), args.learning_rate, weight_decay=0.00005)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5 * args.restart_epochs), int(.75 * args.restart_epochs)])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.restart_epochs)