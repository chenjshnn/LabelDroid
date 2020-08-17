# Generalization and Usefulness Evaludation

### Wilcoxon signed-rank test

To understand the significance of the differences between four kinds of content description, we carry out the Wilcoxon signed-rank test between the scores of our model and each annotator. We then use Benjamini&Hochberg(BH) method to correct p-values for multiple comparisons, and compute effect size r=Z/sqrt(N), where Z is the statisctical result from test and N is the number of observations. The results show that the differences between our model and A1, A2, A3 are mostly significant (p-value<0.05). Detailed results are listed below.

|            | Corrected p-value | Effect size  |
| ---------- | -------- | ------------- | 
| M v.s. A1  | 0.000047 | 0.39  |
| M v.s. A2  | 0.0061   | 0.29  | 
| M v.s. A3  | 0.039    | 0.12  | 
| A3 v.s. A1 | 0.029    | 0.23  | 
| A3 v.s. A2 | 0.0046   | 0.19  | 
| A2 v.s. A1 | 0.77     | 0.086 | 
