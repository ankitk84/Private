from .create_dir import create_dirs
from tqdm import tqdm
import numpy as np
import torch
from torchvision.utils import save_image

import sys
sys.path.append("..")
from metrics.average_meter import AverageMeter
from metrics.calculate_metrics import calculate_metrics

def save_result_Imgs(model, valid_loader, valid_set, device, save_path, thr):
    create_dirs(save_path)
    
    epoch_acc = AverageMeter()
    epoch_recall = AverageMeter()
    epoch_precision = AverageMeter()
    epoch_specificity = AverageMeter()
    epoch_f1_score = AverageMeter()
    epoch_dice = AverageMeter()
    epoch_iou = AverageMeter()
    epoch_auroc = AverageMeter()

    with torch.no_grad():
        model.eval()
        valid_tqdm_batch = tqdm(iterable=valid_loader, total=np.ceil(len(valid_set) / 1))
        c1 = 1
        for images, targets in valid_tqdm_batch:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            preds1 = preds.clone()
            preds1[preds <= thr] = 0
            preds1[preds > thr] = 1

            (acc, recall, specificity, precision,
             f1_score, dice, iou, auroc) = calculate_metrics(preds=preds1, targets=targets, device=device)


            save_image(np.squeeze(images), save_path+'/'+str(c1)+'_images.png')
            save_image(np.squeeze(preds), save_path+'/'+str(c1)+'_preds.png')                    
            save_image(np.squeeze(preds1), save_path+'/'+str(c1)+'_preds1.png')                  
            save_image(np.squeeze(targets), save_path+'/'+str(c1)+'_targets.png')
            c1 += 1


            epoch_acc.update(acc)
            epoch_recall.update(recall)
            epoch_precision.update(precision)
            epoch_specificity.update(specificity)
            epoch_f1_score.update(f1_score)
            epoch_dice.update(dice)
            epoch_iou.update(iou)
            epoch_auroc.update(auroc)


    print('preds at {}- acc:{} | recall:{} | spe:{} | pre:{} | f1_score:{} | dice:{} | auroc:{}'
          .format(thr,
                  epoch_acc.val,
                  epoch_recall.val,
                  epoch_specificity.val,
                  epoch_precision.val,
                  epoch_f1_score.val,
                  epoch_dice.val,
                  epoch_auroc.val))
