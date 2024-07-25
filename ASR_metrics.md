# ASR Metrics
<p style="text-align: center;">What they are and what they mean</p>

### Preface
ASR word segmentation has some definitions that need to be explained:
- Nref is the total number of boundaries in the reference.
- Nf is the total number of detected boundaries.
- Nhit is the number of boundaries correctly detected.

To determine the number of hits, deletions, and insertions, a 20ms tolerance window is added to the reference boundaries.
### Precision
A metric that measures how often a machine learning model correctly predicts the positive class. Precision answers the question: how often is the positive predictions correct?
**In ASR** Precision describes the likelihood of how often the algorithm identifies a correct boundary whenever a boundary is detected.
**Pros:**
- It works well for problems with imbalanced classes since it shows the model correctness in identifying the target class.
- Precision is useful when the cost of a false positive is high. In this case, you typically want to be confident in identifying the target class, even if you miss out on some (or many) instances.

**Cons:**
- Precision does not consider false negatives. Meaning: it does not account for the cases when we miss our target event!
##### Calculation
<span style="color:blue;">Precision = TP/(TP+FP)</span>
The number of correct positive predictions (true positives) divided by the total number of instances the model predicted as positive (both true and false positives).
<span style="color:blue;">Precision_ASR = Nhit/Nf</span>
The higher the precision, the better.

### Recall
A metric that measures how often a machine learning model correctly identifies positive instances (true positives) from all the actual positive samples in the dataset. Recall answers the question: can an ML model find all instances of the positive class?
**In ASR** Recall meansures the overall segmentation accuracy.
**Pros:**
- It works well for problems with imbalanced classes since it is focused on the model’s ability to find objects of the target class.
- Recall is useful when the cost of false negatives is high. In this case, you typically want to find all objects of the target class, even if this results in some false positives (predicting a positive when it is actually negative).

**Cons:**
- Recall does not account for the cost of these false positives.
##### Calculation
<span style="color:blue;">Recall = TP/(TP+FN)</span>
The number of true positives divided by the number of positive instances. The latter includes true positives (successfully identified cases) and false negative results (missed cases).
<span style="color:blue;">Recall_ASR = Nhit/Nref\*100</span>
The higher the recall, the better.

### F1-Score
A metric that measures the harmonic mean of Precision and Recall. Precision and Recall give useful information on their own but they have limitations when viewed separately. Instead of balancing precision and recall, we can just aim for a good F1-score, which would also indicate good Precision and a good Recall value (to balance the precision-recall trade-off). F1-score takes into account both Precision and Recall while avoiding the overestimation that the arithmetic mean might cause.
##### Calculation
<span style="color:blue;">F1 = 2\*Precision\*Recall/(Precision+Recall)</span>, the same definition applies for ASR. The higher the F1-score, the better.

### Over-Segmentation (OS)
A metric that meansures how many fewer/more boundaries are proposed compared to the ground truth. The ratio of the total number of detected boundaries Nf to the number of boundaries in the reference Nref. 
##### Calculation
<span style="color:blue;">OS = (Nf/Nref - 1)\*100</span>
The lower the OS, the better.

### Hit-Rate (HR)
A metric that meansures the overall segmentation accuracy. The ratio of the number of boundaries correctly detected Nhit to the number of boundaries in the reference Nref. 
##### Calculation
<span style="color:blue;">HR = Nhit/Nref\*100</span>
The higher the HR, the better.

### R-Value
F1 is not always sensitive enough to the trade-off between Recall (HR) and OS. R-value is a metric that measures the Recall (HR) and OS scores. OS can boost a lot of metrics by simply adding random boundaries a better HR can be achieved (HR as a function of OS). The R-value indicates how close (distance metric) a segmentation algorithm’s performance is to an ideal point of operation (100% HR with 0% OS). 
The R-value uses two distances in its calculation:
- on the segmentation performance plane, a distance r1 
- the value of under-segmentation compared to over-segmentation is captures in a distance r2.

r1 and r2 are added together and normalized to have a maximum value of 1 at the target-point. R-value not only measures the quality of a segmentation algorithm but can also be used to automatically direct the automatic segmentation process towards a goal.
Here is a [link](https://www.isca-archive.org/interspeech_2009/rasanen09b_interspeech.pdf) to the proposing paper.

##### Calculation
<span style="color:blue;">r1 = sqrt((100-HR)^2+OS^2)</span>
<span style="color:blue;">r2 = (-OS+HR-100)/sqrt(2)</span>
<span style="color:blue;">R-value = 1 - (|r1|+|r2|)/200</span>
The higher the R-value, the better.

### Normalized Edit Distance (NED)
NED is a metric that calculates the dissimilarity of two words also known as the Levenshtein distance. It uses the following ideas:
- Substitution (SUB): occurs when a word gets replaced (for example, “noose” is transcribed as “moose”)
- Insertion (INS): is when a word is added that wasn’t said (for example, “SAT” becomes “essay tea”)
- Deletion (DEL): happens when a word is left out of the transcript completely (for example, “turn it around” becomes “turn around”)
- The number of words in the reference (N)

Uses a DP algorithm similar to the Viterbi algorithm to track the best path from the predicted word to the target word. Here is a [link](https://www.youtube.com/watch?v=C2cRO9BqlZw&list=PLmZlBIcArwhOqEQwyk2TBHmtEKTGPMu5d) to H. Kamper's explanation.
The lower the NED, the better.

### Word Error Rate (WER)
WER is a metric that calculates how much the word string returned by the
recognizer (the hypothesized word string) differs from a reference transcription. It divides the number of errors by the number of words. This metric supports the same ideas as [NED](#normalized-edit-distance-ned) above. 
The first step in computing word error is to compute the minimum edit distance in
words between the hypothesized and correct strings. Then using the minimum substitutions, insertions, and deletions the WER can be calculated.

##### Calculation
<span style="color:blue;">WER = (SUB+INS+DEL)/N</span>
The lower the WER, the better.