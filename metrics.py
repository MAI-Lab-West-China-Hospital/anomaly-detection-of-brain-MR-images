# cal auroc, auprc, sensitivity, |dice|

from sklearn.metrics import precision_recall_curve
import numpy as np
from glob import glob
import os
import pandas as pd
import nibabel as nib
from sklearn.metrics import roc_curve, auc


def get_label(labelarr):
    label_orig = []
    for idx in range(labelarr.shape[-1]):
        if labelarr[:, :, idx].any():
            label_orig.append(1)
        else:
            label_orig.append(0)
    label_after = 1 - np.array(label_orig) + 1

    return label_orig, label_after


def RocPrc(error_map, gt):
    gt_slices = []
    score_slices = []
    for error, gt in zip(error_map, gt):

        gt_ = nib.load(gt).get_fdata()
        error_ = nib.load(error).get_fdata()

        gt_orig, gt_after = get_label(gt_)
        gt_slices.append(gt_after)

        score_ = []

        for idx in range(error_.shape[-1]):
            slice_error = np.sum(error_[:, :, idx])
            score_.append(slice_error)

        score_use = (score_ - np.min(score_)) / (np.max(score_) - np.min(score_))
        score_use = 1 - score_use
        score_slices.append(score_use)

    score_slices = np.concatenate(score_slices, axis=0)
    gt_slices = np.concatenate(gt_slices, axis=0)

    fpr, tpr, thresholds = roc_curve(gt_slices, score_slices, pos_label=2)
    auroc = auc(fpr, tpr)
    precision, recall, th = precision_recall_curve(gt_slices, score_slices, pos_label=2)
    auprc = auc(recall, precision)

    return round(auroc, 4), round(auprc, 4)


def Sens(error_map, gt):
    gt_slices = []
    pred_slices = []

    for error, gt in zip(error_map, gt):

        gt_ = nib.load(gt).get_fdata()
        error_ = nib.load(error).get_fdata()

        for i in range(gt_.shape[-1]):
            gt_slice = gt_[..., i]
            pred_slice = error_[..., i]
            overlap = sum(pred_slice[np.where(gt_slice == 1)])

            if overlap:
                pred_slices.append(1)
            else:
                pred_slices.append(0)

            if gt_slice.any():
                gt_slices.append(1)
            else:
                gt_slices.append(0)

    sens = sum(pred_slices) / sum(gt_slices)
    return round(sens, 4)


def binary_dice(s, g):
    """
    calculate the Dice score of two N-d volumes.
    s: the segmentation volume of numpy array
    g: the ground truth volume of numpy array
    """
    assert (len(s.shape) == len(g.shape))
    prod = np.multiply(s, g)
    s0 = prod.sum()
    dice = (2.0 * s0 + 1e-10) / (s.sum() + g.sum() + 1e-10)
    return round(dice, 4)


def cal_dice(binary_seg, gt, savedir):
    dices = []
    imagenames = []
    for seg, gt in zip(binary_seg, gt):
        imagenames.append(os.path.basename(seg))
        gt_ = nib.load(gt).get_fdata()  # [160, 192, 96]
        seg_ = nib.load(seg).get_fdata()

        dice = binary_dice(seg_, gt_)
        dices.append(dice)

    df = pd.DataFrame(dices, index=imagenames)
    df.to_csv(savedir)
    return np.mean(dices), max(dices)


if __name__ == '__main__':
    # load dataset
    img_dir = '../data/IXI_Brats_t2/register1mm/BraTs19T2'
    recon_dir = '../TorchResults/z512_denseAE/recon_image'
    error_dir = '../TorchResults/z512_denseAE/error_post'
    seg_dir = '../results/z512_denseAE/error_binpost'
    gt_dir = '../data/IXI_Brats_t2/register1mm/BraTs19T2segbin'

    dicesavedir = '../TorchResults/z512_denseAE/dice.csv'

    error_map = sorted(glob(error_dir + "/*.gz"))
    seg = []
    gt = []
    for img in error_map:
        imgname = os.path.basename(img)
        img2segname = imgname.replace("t2", "seg")
        gtname = os.path.join(gt_dir, img2segname)
        gt.append(gtname)
        seg.append(os.path.join(seg_dir, imgname))

    sens = Sens(seg, gt)
    auroc, auprc = RocPrc(error_map, gt)
    dice, max_dice = cal_dice(seg, gt, dicesavedir)
    print(auroc, auprc, sens, dice, max_dice)
