# Unblind Your Apps: Predicting Natural-Language Labels for Mobile GUI Components by Deep Learning

## INTRODUCTION

<b> This image shows an example of UI components and labels. For example, the content description for the top-right image-based button of the UI screenshot is ''more options''.</b>

<img src="./Introduction/Figure1.png" alt="Example of UI components and labels"  width="600"/>

<img src="./Introduction/Figure2.png" alt="Source code for setting up labels for 'add playlist' button (which is indeed a clickable ImageView)"   width="600"/>

<img src="./Introduction/Figure3.png" alt="Examples of image-based buttons 1:clickable ImageView; 2/3:ImageButton"   width="600"/>

## MOTIVATIONAL MINING STUDY

<b>We conduct a motivational mining study of 15,087 apps. Among these apps, we collected 394,489 GUI screenshots, and 70.53% of them contain image-based buttons.</b>

<img src="./Motivational_mining_study/Table1.png" alt="Statistics of label missing situation"  width="600" />

<img src="./Motivational_mining_study/Figure4.png" alt="The distribution of the category of applications with different rate of image-based buttons missing content description"   width="600"/>

<img src="./Motivational_mining_study/Figure5.png" alt="Box-plot for missing rate distribution of all apps with different download number"   width="600"/>

## APPROACH

<img src="./Approach/Figure6.png" alt="Overview of our approach" />


## DATA PREPROCESSING

Preprocessing:
1. Filter duplicate xml
2. Filter duplicate elements by comparing both their screenshots and the content descriptions
3. Filter low-quality labels
(1) Labels contain the class of elments, e.g, "ImageView"
(2) Labels contain the app's name, e.g., "ringtone maker" for App Ringtone Maker
(3) Unfinished labels, e.g., "content description"

The whole list of meaningless labels can be seen at [Data_preprocessing/missing_label.txt](./Dataset/meaningless_label.txt)


## DATASET

<img src="./Dataset/Figure7.png" alt="Examples of our dataset" width="500"/>

<b>Statistics of our dataset:</b>

<img src="./Dataset/Table2.png" alt="Dataset Statistics" width="500"/>

*Dataset can be download at <https://drive.google.com/open?id=18BV1oDsvEVY1xvefLe0QpGBPgpvNGY43>*

## EVALUATION
We evaluate our model in three aspects, i.e., accuracy with
automated testing, generality and usefulness with user study. We
also shows the practical value of LabelDroid by submitting the
labels to app development teams.

### Accuracy

Overall accuracy results
![Accuracy Results](Accuracy/Table3.png)

Results by different category
![Accuracy Results](Accuracy/Figure8.png)

Qualitative Performance with Baselines
![Accuracy Results](Accuracy/Table4.png)

Common causes for generation failure
![Accuracy Results](Accuracy/Table5.png)


### Generalization & Usefulness

App details and results:
<img src="Generalization&Usefulness/app_details.png" alt="The acceptability score (AS) and the standard deviation for 12 completely unseen apps. * denotes p < 0.05."/>

<img src="Generalization&Usefulness/boxplot.png" alt="Distribution of app acceptability scores by human
annotators (A1, A2, A3) and the model (M)" width="500"/>

<img src="Generalization&Usefulness/Table7.png" alt="Examples of generalization" />

[Generalization&Usefulness](https://github.com/icse2020Accessibility/icse2020Accessibility/blob/master/Generalization%26Usefulness) contains all data we used in this part and the results from model(M), developers(A1,A2,A3) and Evaluator.
