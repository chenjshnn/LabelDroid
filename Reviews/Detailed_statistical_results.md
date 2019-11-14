# Accuracy

### (Section 6.3.1) The significance of LabelDroid compared with other baselines -- Wilcoxon rank sum test for accuracy evaluation 

To evalute whether the differences are significant, we further conduct the <b>Wilcoxon rank-sum test</b> between LabelDroid and CNN+LSTM and CNN+CNN respectively in all testing metrics (BLEU@1,2,3,4, METEOR, ROUGE-L, CIDEr). We then use Benjamini&Hochberg(BH) method to correct p-values for multiple comparisons. Results show that <b>the improvement of our model is significant in all comparisons</b>, and the detailed corrected p-values are listed below.

|                | BLEU@1 | BLEU@2  | BLEU@3 | BLEU@4 | METEOR  | ROUGE-L | CIDEr |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| LabelDroid v.s. CNN+LSTM  | 0.0017 | 0.0023  | 0.0061 | 0.0062 | 0.00093 | 0.00097 | 0.0014 |
| LabelDroid v.s. CNN+CNN   | 0.0017 | 0.00096 | 0.0033 | 0.0029 | 0.00083 | 0.00093 | 0.0014 |
| CNN+LSTM v.s. CNN+CNN     | 0.88   | 0.62    | 0.66   | 0.63   | 0.83    | 0.85    | 0.88   |


# Generalization and Usefulness Evaludation

### (Section 6.4.2) The significant of LabelDroid compared with three human annotators -- Wilcoxon signed-rank test

To understand the significance of the differences between four kinds of content descriptions, we carry out the Wilcoxon signed-rank test between the scores of our model and each annotator. We then use Benjamini&Hochberg(BH) method to correct p-values for multiple comparisons, and compute effect size r=Z/sqrt(N), where Z is the statisctical result from test and N is the number of observations. The results show that the differences between our model and A1, A2, A3 are mostly significant (p-value<0.05). Detailed results are listed below.

|            | Corrected p-value | Effect size  |
| ---------- | -------- | ------------- | 
| M v.s. A1  | 0.000047 | 0.39  |
| M v.s. A2  | 0.0061   | 0.29  | 
| M v.s. A3  | 0.039    | 0.12  | 
| A3 v.s. A1 | 0.029    | 0.23  | 
| A3 v.s. A2 | 0.0046   | 0.19  | 
| A2 v.s. A1 | 0.77     | 0.086 | 
