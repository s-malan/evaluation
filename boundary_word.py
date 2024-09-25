"""
Funtions used to evaluate word segmentation algorithm. Called as main script, it evaluates the input segmentation file.

Author: Herman Kamper, Simon Malan
Contact: kamperh@gmail.com, 24227013@sun.ac.za
Date: March 2024
"""

import numpy as np
import argparse
from pathlib import Path
import textgrids
from tqdm import tqdm
import itertools

def eval_segmentation(seg, ref, strict=True, tolerance=1, continuous=False, num_seg=None, num_ref=None, num_hit=None):
    """
    Calculate number of hits of the segmentation boundaries with the ground truth boundaries.

    Parameters
    ----------
    seg : list of list of int
        The segmentation hypothesis word boundary frames for all utterances in the sample.
    ref : list of list of int
        The ground truth reference word boundary frames for all utterances in the sample.
    tolerance : int
        The number of offset frames that a segmentation hypothesis boundary can have with regards to a reference boundary and still be regarded as correct.
        default: 1 (10ms for MFCCs with a frame shift of 10ms or or 20ms for self-supervised speech models)
    continuous : bool
        If True, return the number of segments, references, and hits instead of the evaluation metrics. This is to continue the evaluation over multiple samples.
        default: False
    num_seg : int
        The current number of segmentation boundaries.
        default: None
    num_ref : int
        The current number of reference boundaries.
        default: None
    num_hit : int
        The current number of hits.
        default: None

    Return
    ------
    output : (float, float, float)
        precision, recall, F-score.
    
    or if continuous:
        output : (int, int, int)
            num_seg, num_ref, num_hit
    """

    if not continuous:
        num_seg = 0 #Nf
        num_ref = 0 #Nref
        num_hit = 0 #Nhit
    
    assert len(seg) == len(ref) # Check if the number of utterances in the hypothesis and reference are the same
    for i_utterance in range(len(seg)): # for each utterance
        prediction = seg[i_utterance]
        ground_truth = ref[i_utterance]

        if len(prediction) > 0 and len(ground_truth) > 0 and abs(prediction[-1] - ground_truth[-1]) <= tolerance: # if the last boundary is within the tolerance, delete it since it would have hit
            prediction = prediction[:-1]
            if len(ground_truth) > 0: # Remove the last boundary of the reference if there is more than one boundary
                ground_truth = ground_truth[:-1] 
        # this helps when the segmentation algo does not automatically predict a boundary at the end of the utterance

        num_seg += len(prediction)
        num_ref += len(ground_truth)

        if len(prediction) == 0 or len(ground_truth) == 0: # no hits possible
            continue

        # count the number of hits
        for i_ref in ground_truth:
            for i, i_seg in enumerate(prediction):
                if abs(i_ref - i_seg) <= tolerance:
                    num_hit += 1
                    prediction.pop(i) # remove the segmentation boundary that was hit
                    if strict: break # makes the evaluation strict, so that a reference boundary can only be hit once

    # Return the current counts
    return num_seg, num_ref, num_hit
    
def get_p_r_f1(num_seg, num_ref, num_hit):
    """
    Calculate precision, recall, F-score for the segmentation boundaries.

    Parameters
    ----------
    num_seg : int
        The current number of segmentation boundaries.
        default: None
    num_ref : int
        The current number of reference boundaries.
        default: None
    num_hit : int
        The current number of hits.
        default: None

    Return
    ------
    output : (float, float, float)
        precision, recall, F-score.
    """

    # Calculate metrics, avoid division by zero:
    if num_seg == num_ref == 0:
        return 0, 0, -np.inf
    elif num_hit == 0:
        return 0, 0, 0
    
    if num_seg != 0:
        precision = float(num_hit/num_seg)
    else:
        precision = np.inf
    
    if num_ref != 0:
        recall = float(num_hit/num_ref)
    else:
        recall = np.inf
    
    if precision + recall != 0:
        f1_score = 2*precision*recall/(precision+recall)
    else:
        f1_score = -np.inf
    
    return precision, recall, f1_score

def get_os(precision, recall):
    """
    Calculates the over-segmentation; how many fewer/more boundaries are proposed compared to the ground truth.

    Parameters
    ----------
    precision : float
        How often word segmentation correctly predicts a word boundary.
    recall : float
        How often word segmentation's prediction matches a ground truth word boundary.

    Return
    ------
    output : float
        over-segmentation
    """

    if precision == 0:
        return -np.inf
    else:
        return recall/precision - 1
    
def get_rvalue(precision, recall):
    """
    Calculates the R-value; indicates how close (distance metric) the word segmentation performance is to an ideal point of operation (100% HR with 0% OS).

    Parameters
    ----------
    precision : float
        How often word segmentation correctly predicts a word boundary.
    recall : float
        How often word segmentation's prediction matches a ground truth word boundary.

    Return
    ------
    output : float
        R-Value
    """

    os = get_os(precision, recall)
    r1 = np.sqrt((1 - recall)**2 + os**2)
    r2 = (-os + recall - 1)/np.sqrt(2)

    rvalue = 1 - (np.abs(r1) + np.abs(r2))/2
    return rvalue

def get_word_token_boundaries(seg, ref, tolerance=1):
    """
    Calculate precision, recall, F-score for the word token boundaries.

    Parameters
    ----------
    ref : list of vector of bool
        The ground truth reference.
    seg : list of vector of bool
        The segmentation hypothesis.
    tolerance : int
        The number of slices with which a boundary might differ but still be
        regarded as correct.
        default: 1

    Return
    ------
    output : (float, float, float)
        Precision, recall, F-score.
    """

    n_tokens_ref = 0
    n_tokens_seg = 0
    n_tokens_correct = 0

    assert len(seg) == len(ref)
    
    for i_utterance in range(len(seg)): # for each utterance
        prediction = seg[i_utterance]
        ground_truth = ref[i_utterance]

        seg_intervals = [(a,b) for a,b in itertools.pairwise([0] + prediction)]
        ref_intervals = [(a,b) for a,b in itertools.pairwise([0] + ground_truth)]

        # Build list of ((word_start_lower, word_start_upper), (word_end_lower, word_end_upper))
        word_bound_intervals = []
        for word_start, word_end in ref_intervals:
            word_bound_intervals.append((
                (max(0, word_start - tolerance), word_start + tolerance),
                (word_end - tolerance, word_end + tolerance)
                ))
        
        n_tokens_ref += len(word_bound_intervals)
        n_tokens_seg += len(seg_intervals)

        # Score word token boundaries
        for seg_start, seg_end in seg_intervals:
            for i_gt_word, (word_start_interval, word_end_interval) in enumerate(word_bound_intervals):
                word_start_lower, word_start_upper = word_start_interval
                word_end_lower, word_end_upper = word_end_interval

                if (word_start_lower <= seg_start <= word_start_upper and word_end_lower <= seg_end <= word_end_upper):
                    n_tokens_correct += 1
                    word_bound_intervals.pop(i_gt_word)  # can't re-use token
                    break

    precision = float(n_tokens_correct)/n_tokens_seg
    recall = float(n_tokens_correct)/n_tokens_ref
    if precision + recall != 0:
        f = 2*precision*recall / (precision + recall)
    else:
        f = -np.inf

    return precision, recall, f

def get_frame_num(seconds, frames_per_ms=20):
        """
        Convert seconds to feature embedding frame number

        Parameters
        ----------
        seconds : float or ndarray (float)
            The number of seconds (of audio) to convert to frames

        Return
        ------
        output : int
            The feature embedding frame number corresponding to the given number of seconds 
        """

        return np.round(seconds / frames_per_ms * 1000).astype(np.int16) # seconds (= samples / sample_rate) / x ms per frame * 1000ms per second

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "disc_path",
        metavar="disc-path",
        help="path to the discovered fragments.",
        type=Path,
    )
    parser.add_argument(
        "gold_dir",
        metavar="gold-dir",
        help="path to the directory of alignments.",
        type=Path,
    )
    parser.add_argument(
        "--alignment_format",
        metavar="--alignment-format",
        help="extension of the alignment files.",
        default=".TextGrid",
        type=str,
    )
    parser.add_argument(
        "--frames_per_ms",
        metavar="--frames-per-ms",
        help="number of ms in a frame for the encoding.",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--tolerance",
        help="the tolerance in number of frames.",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--strict",
        help="optional variable to follow strict evaluation.",
        default=True,
        type=bool,
    )
    args = parser.parse_args()

    files_seg = args.disc_path.rglob("**/*" + ".list")
    seg_list = []
    ref_list = []
    for file_seg in tqdm(files_seg):
        boundaries = []
        references = []
        with open(file_seg, "r") as f:
            for line in f:
                if len(line.split(" ")) == 2: # end_time class
                    end_time, _ = line.split(" ")
                    boundaries.append(float(end_time))
                else: # end_time
                    boundaries.append(float(line))
        seg_list.append(list(get_frame_num(np.array(boundaries), frames_per_ms=args.frames_per_ms))) # convert to frames

        file_ref = list(args.gold_dir.rglob(f'**/{file_seg.stem}' + args.alignment_format))[0]
        if args.alignment_format == '.TextGrid':
            for word in textgrids.TextGrid(file_ref)['words']:
                references.append(float(word.xmax))
        elif args.alignment_format == '.txt':
            with open(file_ref, 'r') as f:
                for line in f:
                    line = line.split()
                    references.append(float(line[1]))
        ref_list.append(list(get_frame_num(np.array(references), frames_per_ms=args.frames_per_ms)))

    # evaluate the segmentation
    num_seg, num_ref, num_hit = eval_segmentation(seg_list, ref_list, strict=args.strict, tolerance=args.tolerance)
    precision, recall, f1_score = get_p_r_f1(num_seg, num_ref, num_hit)
    os = get_os(precision, recall)
    rvalue = get_rvalue(precision, recall)
    token_p, token_r, token_f1 = get_word_token_boundaries(seg_list, ref_list, tolerance=args.tolerance)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1_score}")
    print(f"Over-segmentation: {os}")
    print(f"R-value: {rvalue}")
    print(f"Token Precision: {token_p}")
    print(f"Token Recall: {token_r}")
    print(f"Token F1-score: {token_f1}")