# For Reviewer_A

#### Q: Where are the 12 apps in Section6.4 from?*

A: In Section 6.4, we randomly select 12 mobile apps from the data in empirical study for human evaluation. Note that these 12 mobile apps are not in training/validation/testing sets, as these apps have NONE labeled image-based buttons to avoid potential bias, and we only consider apps which have at least 1M installations and at least 15 screenshots to avoid bias. We adopt our model and human annotators to generate the labels for image-based buttons and ask the human evaluator to judge their quality.

#### Q: Does the rater catch all deliberate inserted lables?*

A: Yes. In Section 6.4.1, we deliberately inserting 2 wrong labels and 2 suitable labels. The evaluator successfully identifies them, showing his capability in the label judge.


# For Reviewer_B

## Q: Detailed descriptions of our dataset?

A: There are 2513 unique labels which appear at least five times in our dataset. The label <u>Download</u> appears 110 times, and <u>Next</u> appears 170 times. To render an overview of our dataset, we draw a word cloud as seen in [Word_Cloud](https://github.com/icse2020Accessibility/icse2020Accessibility/blob/master/Dataset/wordcloud.png) 

## Q: Suggest multiple rounds of validation?*

A: Thanks for the suggestion. Due to the limited time for rebuttal and the long training time of our model (over 17 hours), we could not provide the results for multiple rounds of validation right now. We would add the results to Section 6.3 in the revised paper in the future. 

## Q: May add Thread to Validty?*
A: We will add this part in revision. 


# For Reviewer_C

## Q: 77% number on the summary of Section3.2*
A: It is in fact 77.38% mentioned in table1, which indicates that 77.38% of 10,408 apps, which contain image-based buttons, have at least one button lacking labels.

