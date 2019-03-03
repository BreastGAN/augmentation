import glob
import numpy as np
import sys

from models.rcnn.eval import print_evaluation_scores

# pattern = '/scratch_net/biwidl104/jendelel/mammography/data_out/FRCNN_split_gen*/outputs*.json'
pattern = sys.argv[1]  # '/scratch_net/biwidl104/jendelel/mammography/data_out/FRCNN_split_augm*/outputs*.json'
print("File pattern: {}".format(pattern), flush=True)
all_results = []
for conf_score in np.linspace(0.1, 0.9):
    for fname in glob.glob(pattern):
        results = print_evaluation_scores(fname, confidence_score=conf_score)
        auc = results['roc_auc']
        fpr, tpr, thresholds = results['roc']
        f1 = results['f1']
        fooled = results["fooling/fooled"]
        all_results.append({"file": fname, "conf_score": conf_score, "auc": auc, "f1": f1, "results": results})

best = max(all_results, key=lambda x: x["auc"])
print(best)
