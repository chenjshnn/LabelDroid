# Accuracy

### Wilcoxon rank sum test for accuracy evaluation

To evalute whether the differences are significant, we further conduct the <b>Wilcoxon rank-sum test</b> between LabelDroid and CNN+LSTM and CNN+CNN respectively in all testing metrics (BLEU@1,2,3,4, METEOR, ROUGE-L, CIDEr). We then use Benjamini&Hochberg(BH) method to correct p-values for multiple comparisons. Results show that <b>the improvement of our model is significant in all comparisons</b>, and the detailed corrected p-values are listed below.

|                | BLEU@1 | BLEU@2  | BLEU@3 | BLEU@4 | METEOR  | ROUGE-L | CIDEr |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| LabelDroid v.s. CNN+LSTM  | 0.0017 | 0.0023  | 0.0061 | 0.0062 | 0.00093 | 0.00097 | 0.0014 |
| LabelDroid v.s. CNN+CNN   | 0.0017 | 0.00096 | 0.0033 | 0.0029 | 0.00083 | 0.00093 | 0.0014 |
| CNN+LSTM v.s. CNN+CNN     | 0.88   | 0.62    | 0.66   | 0.63   | 0.83    | 0.85    | 0.88   |
