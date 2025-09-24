import numpy as np
import torch
import os
import traceback
import time
import nrrd
import sys
import matplotlib.pyplot as plt
import logging
import argparse
import torch.nn.functional as F
from scipy.stats import norm
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parallel.data_parallel import data_parallel
from scipy.ndimage.measurements import label
from scipy.ndimage import center_of_mass
from net.sanet import SANet
from dataset.collate import train_collate, test_collate, eval_collate
from dataset.bbox_reader import BboxReader
from config import config
import pandas as pd
from evaluationScript.noduleCADEvaluationLUNA16 import noduleCADEvaluation

this_module = sys.modules[__name__]
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--net', '-m', metavar='NET', default=config['net'],
                    help='neural net')
parser.add_argument("--mode", type=str, default = 'eval',
                    help="you want to test or val")
parser.add_argument("--weight", type=str, default='./results/model/model.ckpt',
                    help="path to model weights to be used")
parser.add_argument("--dicom-path", type=str, default=None,
                    help="path to dicom files of patient")
parser.add_argument("--out-dir", type=str, default=config['out_dir'],
                    help="path to save the results")
parser.add_argument("--test-set-name", type=str, default=config['test_set_name'],
                    help="path to save the results")
parser.add_argument("--fold-num", type=str, default='1',
                    help="fold-num")
parser.add_argument("--preprocessed-data-dir", type=str, default='1',
                    help="data path")


def main():
    logging.basicConfig(format='[%(levelname)s][%(asctime)s] %(message)s', level=logging.INFO)
    args = parser.parse_args()

    if args.mode == 'eval':
        data_dir = args.preprocessed_data_dir
        test_set_name = args.test_set_name
        num_workers = 16
        initial_checkpoint = args.weight
        net = args.net
        out_dir = args.out_dir

        net = getattr(this_module, net)(config)
        net = net.cuda()

        if initial_checkpoint:
            print('[Loading model from %s]' % initial_checkpoint)
            checkpoint = torch.load(initial_checkpoint)
            epoch = checkpoint['epoch']

            net.load_state_dict(checkpoint['state_dict'])
        else:
            print('No model weight file specified')
            return

        print('out_dir', out_dir)
        save_dir = os.path.join(out_dir, 'res', str(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(os.path.join(save_dir, 'FROC')):
            os.makedirs(os.path.join(save_dir, 'FROC'))

        dataset = BboxReader(data_dir, test_set_name, config['val_anno'][0], config, 0, mode='eval')
#         test_loader = DataLoader(dataset, batch_size=1, shuffle=False,
#                                  num_workers=num_workers, pin_memory=False, collate_fn=train_collate)
        eval(net, dataset, save_dir)
        noduleCADEval(dataset, save_dir,epoch,out_dir,args.fold_num)
    else:
        logging.error('Mode %s is not supported' % (args.mode))


def eval(net, dataset, save_dir):
    net.set_mode('eval')
    net.use_rcnn = True

    print('Total # of eval data %d' % (len(dataset)))
    for i, (input, truth_bboxes, truth_labels, crop_size) in enumerate(dataset):
        try:
            input = Variable(input).cuda().unsqueeze(0)
            truth_bboxes = np.array(truth_bboxes)
            truth_labels = np.array(truth_labels)
            pid = dataset.filenames[i]

            print('[%d] Predicting %s' % (i, pid))

            with torch.no_grad():
                net.forward(input, truth_bboxes, truth_labels)

            rpns = net.rpn_proposals.cpu().numpy()
            detections = net.detections.cpu().numpy()
            ensembles = net.ensemble_proposals.cpu().numpy()

#             print('detections', detections.shape)

            if len(rpns):
                rpns = rpns[:, 1:]
                for index in range(len(rpns)):
                    rpns[index][1]+=crop_size[0]
                    rpns[index][2]+=crop_size[1]
                    rpns[index][3]+=crop_size[2]
                np.save(os.path.join(save_dir, '%s_rpns.npy' % (pid)), rpns)

            if len(detections):
                detections = detections[:, 1:-1]
                for index in range(len(detections)):
                    detections[index][1]+=crop_size[0]
                    detections[index][2]+=crop_size[1]
                    detections[index][3]+=crop_size[2]
                np.save(os.path.join(save_dir, '%s_rcnns.npy' % (pid)), detections)

            if len(ensembles):
                ensembles = ensembles[:, 1:]
                for index in range(len(ensembles)):
                    ensembles[index][1]+=crop_size[0]
                    ensembles[index][2]+=crop_size[1]
                    ensembles[index][3]+=crop_size[2]
                np.save(os.path.join(save_dir, '%s_ensembles.npy' % (pid)), ensembles)

            # Clear gpu memory
            del input, truth_bboxes, truth_labels
            torch.cuda.empty_cache()

        except Exception as e:
            del input, truth_bboxes, truth_labels
            torch.cuda.empty_cache()
            traceback.print_exc()

            return
    
    # Generate prediction csv for the use of performning FROC analysis
#     res = []
#     for pid in dataset.filenames:
#         if os.path.exists(os.path.join(save_dir, '%s_detections.npy' % (pid))):
#             detections = np.load(os.path.join(save_dir, '%s_detections.npy' % (pid)))
#             detections = detections[:, [3, 2, 1, 4, 0]]
#             names = np.array([[pid]] * len(detections))
#             res.append(np.concatenate([names, detections], axis=1))
    rpn_res = []
    rcnn_res = []
    ensemble_res = []
    for pid in dataset.filenames:
        if os.path.exists(os.path.join(save_dir, '%s_rpns.npy' % (pid))):
            rpns = np.load(os.path.join(save_dir, '%s_rpns.npy' % (pid)))
            rpns = rpns[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(rpns))
            rpn_res.append(np.concatenate([names, rpns], axis=1))

        if os.path.exists(os.path.join(save_dir, '%s_rcnns.npy' % (pid))):
            rcnns = np.load(os.path.join(save_dir, '%s_rcnns.npy' % (pid)))
            rcnns = rcnns[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(rcnns))
            rcnn_res.append(np.concatenate([names, rcnns], axis=1))

        if os.path.exists(os.path.join(save_dir, '%s_ensembles.npy' % (pid))):
            ensembles = np.load(os.path.join(save_dir, '%s_ensembles.npy' % (pid)))
            ensembles = ensembles[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(ensembles))
            ensemble_res.append(np.concatenate([names, ensembles], axis=1))
    
    rpn_res = np.concatenate(rpn_res, axis=0)
    rcnn_res = np.concatenate(rcnn_res, axis=0)
    ensemble_res = np.concatenate(ensemble_res, axis=0)
    col_names = ['seriesuid','coordX','coordY','coordZ','diameter_mm', 'probability']
    eval_dir = os.path.join(save_dir, 'FROC')
    rpn_submission_path = os.path.join(eval_dir, 'submission_rpn.csv')
    rcnn_submission_path = os.path.join(eval_dir, 'submission_rcnn.csv')
    ensemble_submission_path = os.path.join(eval_dir, 'submission_ensemble.csv')
    
    df = pd.DataFrame(rpn_res, columns=col_names)
    df.to_csv(rpn_submission_path, index=False)

    df = pd.DataFrame(rcnn_res, columns=col_names)
    df.to_csv(rcnn_submission_path, index=False)

    df = pd.DataFrame(ensemble_res, columns=col_names)
    df.to_csv(ensemble_submission_path, index=False)
    
def noduleCADEval(dataset, save_dir,epoch,out_dir,fold_num):
    eval_dir = os.path.join(save_dir, 'FROC')
    rpn_submission_path = os.path.join(eval_dir, 'submission_rpn.csv')
    rcnn_submission_path = os.path.join(eval_dir, 'submission_rcnn.csv')
    ensemble_submission_path = os.path.join(eval_dir, 'submission_ensemble.csv')
    # Start evaluating
    if not os.path.exists(os.path.join(eval_dir, 'rpn')):
        os.makedirs(os.path.join(eval_dir, 'rpn'))
    if not os.path.exists(os.path.join(eval_dir, 'rcnn')):
        os.makedirs(os.path.join(eval_dir, 'rcnn'))
    if not os.path.exists(os.path.join(eval_dir, 'ensemble')):
        os.makedirs(os.path.join(eval_dir, 'ensemble'))
        
    annotations_filename = '/data/ltz-dg/filelist/annotations/annotations-luna16-lungseg.csv'
    val_path = '/data/ltz-dg/filelist/annotations/seriesuids-luna16-lungseg.csv'
    annotation_exclude = '/data/ltz-dg/filelist/annotations/luna16_excluded.csv'
    
    rpn_sens_boot,rpn_froc_boot,rpn_sens_norm,rpn_froc_norm = noduleCADEvaluation(annotations_filename,annotation_exclude,val_path, rpn_submission_path, os.path.join(eval_dir, 'rpn'))

    rcnn_sens_boot,rcnn_froc_boot,rcnn_sens_norm,rcnn_froc_norm = noduleCADEvaluation(annotations_filename,annotation_exclude,val_path, rcnn_submission_path, os.path.join(eval_dir, 'rcnn'))

    ensemble_sens_boot,ensemble_froc_boot,ensemble_sens_norm,ensemble_froc_norm = noduleCADEvaluation(annotations_filename,annotation_exclude,val_path, ensemble_submission_path, os.path.join(eval_dir, 'ensemble'))
    
#     sens_boot,froc_boot,sens_norm,froc_norm =noduleCADEvaluation(annotations_filename,'/root/workspace/imgctl/SANet/evaluationScript/annotations_excluded.csv', val_path, res_path, os.path.join(eval_dir, 'res'))
    
    content=[]
    content.append([str(epoch),'froc(avg)','0.125','0.25','0.5','1','2','4','8'])
    content.append(['rpn_boot']+[rpn_froc_boot]+rpn_sens_boot)
    content.append(['rcnn_boot']+[rcnn_froc_boot]+rcnn_sens_boot)
    content.append(['ensemble_boot']+[ensemble_froc_boot]+ensemble_sens_boot)
    content.append(['rpn_norm']+[rpn_froc_norm]+rpn_sens_norm)
    content.append(['rcnn_norm']+[rcnn_froc_norm]+rcnn_sens_norm)
    content.append(['ensemble_norm']+[ensemble_froc_norm]+ensemble_sens_norm)
    content_dataframe = pd.DataFrame(content)
    types = 'include_fp'
    if not os.path.exists(out_dir+'froc_'+types+'.csv'):
        content_dataframe.to_csv(out_dir+'froc_'+types+'.csv',index=False,header=None)
    else:
        exist_dataframe = pd.read_csv(out_dir+'froc_'+types+'.csv',header=None)
        new_dataframe = pd.concat([exist_dataframe,content_dataframe],axis=0)
        new_dataframe.to_csv(out_dir+'froc_'+types+'.csv',index=False,header=None)
        
    simple = []
    simple.append([str(epoch),rpn_froc_boot,rcnn_froc_boot,ensemble_froc_boot,rpn_froc_norm,rcnn_froc_norm,ensemble_froc_norm])
    simple_dataframe = pd.DataFrame(simple)
    types='simple'
    if not os.path.exists(out_dir+'froc_'+types+'.csv'):
        simple_dataframe.to_csv(out_dir+'froc_'+types+'.csv',index=False,header=['epoch','rpn_boot','rcnn_boot','ensemble_boot','rpn_norm','rcnn_norm','ensemble_norm'])
    else:
        exist_dataframe = pd.read_csv(out_dir+'froc_'+types+'.csv',header=None)
        new_dataframe = pd.concat([exist_dataframe,simple_dataframe],axis=0)
        new_dataframe.to_csv(out_dir+'froc_'+types+'.csv',index=False,header=None)

if __name__ == '__main__':
    main()
