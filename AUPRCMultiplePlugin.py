
# matplotlib
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('ggplot')
import pandas
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

colors = matplotlib.cm.Set1(range(10))


def plot_AUPRC(df, score_name, label_name, model_name, ax, color, reverse_sign=False):
    # Precision-Recall curve
    pred_scores = - df[score_name] if reverse_sign else df[score_name]
    precision, recall, thresholds = precision_recall_curve(df[label_name], pred_scores)
    AP = average_precision_score(df[label_name], pred_scores)
    ax.plot(recall, precision, linestyle='-',color=color, label=f'{model_name} (AP={int(AP*10000)/100})')


import PyIO
import PyPluMA
import pickle

class AUPRCMultiplePlugin:
    def input(self, inputfile):
       self.parameters = PyIO.readParameters(inputfile)
    def run(self):
        pass
    def output(self, outputfile):
        # At the moment this takes four files: Three datasets, and a set of other
        # Future will be to make that part flexible
        fig, ax = plt.subplots()
        if ("file1" in self.parameters):
           scores_df = pandas.read_csv(PyPluMA.prefix()+"/"+self.parameters["file1"])
           plot_AUPRC(scores_df, 'score', 'label', self.parameters["label1"], ax, colors[0], reverse_sign=1)
        if ("file2" in self.parameters):
           SCORES_MASIF=PyPluMA.prefix()+"/"+self.parameters["file2"]#"data/masif_test/MaSIF-Search_scores.csv"
           scores_masif = pandas.read_csv(SCORES_MASIF)
           plot_AUPRC(scores_masif, 'score', 'label', self.parameters["label2"], ax, colors[8], reverse_sign=1)

        if ("file3" in self.parameters):
           dMASIF_SCORES=PyPluMA.prefix()+"/"+self.parameters["file3"]#"data/masif_test/dmasif_out.csv"
           scores_dmasif = pandas.read_csv(dMASIF_SCORES)
           plot_AUPRC(scores_dmasif, 'avg_score', 'label', self.parameters["label3"], ax, colors[1], reverse_sign=0)
           
        
        OTHER_SCORES=PyPluMA.prefix()+"/"+self.parameters["other"]#"data/masif_test/Other_tools_SCORES.csv"

        #scores_df = pandas.read_csv("PIsToN_scores.csv")
        other_labels = PyIO.readSequential(PyPluMA.prefix()+"/"+self.parameters["other_labels"])
        if (OTHER_SCORES.endswith('csv')):
           scores_other = pandas.read_csv(OTHER_SCORES)
        else:
           oscore = open(OTHER_SCORES, "rb")
           scores_other = pickle.load(oscore)
           for mylabel in other_labels:
              scores_other[mylabel] = scores_other[mylabel].astype(float)

        #other_labels = ['FIREDOCK', 'AP_PISA', 'CP_PIE', 'PYDOCK_TOT', 'ZRANK2', 'ROSETTADOCK', 'SIPPER']
        #pos_label = [0,0,1,0,0,0,1]
        reverse_sign = PyIO.readSequential(PyPluMA.prefix()+"/"+self.parameters["pos_label"])
        #reverse_sign = [1,1,0,1,1,1,0]
        #colors_array = ['r', 'c', 'm', 'y', 'black', 'orange', 'tan']
        for i in range(len(other_labels)):
          plot_AUPRC(scores_other, other_labels[i], 'Label', other_labels[i], ax, colors[i+2], reverse_sign=reverse_sign[i])



        # # title
        plt.title('Precision-Recall curves')
        # x label
        plt.xlabel('Recall')
        # y label
        plt.ylabel('Precision')

        plt.legend(loc='best')
        plt.savefig(outputfile,dpi=600)
        plt.show();

