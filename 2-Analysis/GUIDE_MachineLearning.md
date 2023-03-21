# Machine learning decoding
Multivariate pattern analysis and classification

## Intro  
The MNE-toolbox for EEG/MEG is a great option to apply MVPA and machine learning classification (using Scikit-learn libs)

*Additional documentation*

Tutorial: 

https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html#sphx-glr-auto-tutorials-machine-learning-50-decoding-py

Read this for more theoretical input on MVPA approach in MNE: King et al., 2018, https://hal.archives-ouvertes.fr/hal-01848442.

*Examples* 

- Example 1. Simple to follow example of classification. https://natmeg.se/mne_multivariate/mne_multivariate.html

- Example 2. MVPA in infant data. https://github.com/BayetLab/infant-EEG-MVPA-tutorial

- Example 3. Time-resolved MVPA decoding two tasks (Marti et al., 2015; https://doi.org/10.1016/j.neuron.2015.10.040)

- Example 4. Peer et al., 2017 EEG analysis of novelty and pleasantness of stimulki (published in _plos One_) https://neuro.inf.unibe.ch/AlgorithmsNeuroscience/Tutorial_files/DatasetConstruction.html

When classifying EEG data we may choose: 
* Which algorithm do we use ? 
* Do we apply it in space (e.g., taking the entire epoch) or in time (at each time point)? 
* Do we use apply it at single-subject or at group level ? 


## Data preparation
### Labeling 
We first need to make sure the epochs are correctly label according to our research question (e.g., We may have trials with different noise conditions and but be interested in labeling them only on whether they were correct/incorrect). Event labels can be manipulated in MNE toolbox using dictionaries in the epochs.events and epochs.event_id fields (mne Epoch object). 
Of course, only the epochs and channels of interest should be also be passed to the classifier. 

### Transformations 
Transformations like filters can be applied depending on your features of interest.
See MNE documentation: https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html
and Scikit-learn: https://scikit-learn.org/stable/data_transforms.html/ 

##### Scaling
Some studies, mainly focused on topographies (amplitudes of all electrodes) on a post stimuli period suggest removing mean and scaling the features (channels) to unit variance. That is , z= (x-u)/s where u is mean and s standard deviation. This is done to prevent some channels, e.g., with larger variability to dominate the model. 

To *scale* each *channel* with mean and sd computed accross of all its time points and epochs can be done with the [mne.decoding.Scaler](https://mne.tools/dev/generated/mne.decoding.Scaler.html). This is different from scikit-learn scalers like [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler), which scales the *classification features* (i.e., each time point for each channel by estimating using mean and sd using data from all epochs).  


##### Vectorizer 
While scikit-learn transformers and estimators usually expect 2D data MNE transformers usually output data with more dimensions. A *vectorizer* is usually applied between MNE and scikit learn steps.[mne.decoding.Vectorizer](https://mne.tools/dev/generated/mne.decoding.Vectorizer.html) transforms n-dimensional arrays into 2D arrays of n_samples by n_features. The result is an array that can be used by the estimators and transformers of scikit-learn. The original shape of the data is saved in the result (attribute: features_shape_) 

For example: 
`clf = make_pipeline(SpatialFilter(), _XdawnTransformer(), Vectorizer(),LogisticRegression())' 


## Scikit-learn 
### Main elements
See : https://scikit-learn.org/stable/developers/develop.html#:~:text=An%20estimator%20is%20an%20object,estimator.
See also the glossary section of scikit-learn https://scikit-learn.org/stable/glossary

#### Estimator
The predominant object in the the API. This is an object that fits a model based on some training data and can inferr some properties on new data. It can be e.g., a classifier or a regressor. All estimators implement the `fit` method: `estimator.fit(X,y)` 

The model is estimated as a deterministic function of:
- Parameters (can be provided with set_params)
- Global numpy.random random state if the estimator's random_state parameters is set to None
- data or sample properties passed to the most recent call to fit, fit_transform or fit_predict or ina a sequence of partial_fit call.

They must provide a `fit` method  and provide set_params and get_params. 

#### Fitting 
The `fit()` method implements estimation of some parameters in the model. 
**Parameters
 - X: array-like of shape (n_samples,n_features)
 - y: array-like of shape (n_samples)
 
 X.shape[0] should be the same as y.shape[0]
 
 *E.g., in EEG MVPA decoding using the entire epochs, X would be of shape (n_epochs, n_channels x n_data-points), 'y' would be of shape (n_epochs) with the label of each epoch* 
 
#### Feature extractors 
A transformer takes input and produce an array-like object of features for each sample (a 2D array-like for a set of samples). They must implement at least: 

- fit. This method is provided on every estimator. Usually takes samples X and targets y
- transform
- get_feature_names_out 
 
### Pipelines
[Sklearn pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) can be used to build a chain of transforms and estimators.
The steps in the function are defined in the order of execution. For instance, if we want to use an  SVM classifier but we need to vectorize and scale the data  before that , our function could be something like this : 

`clf_svm_0 = make_pipeline(Vectorizer(), StandardScaler(), svm.SVC(kernel='rbf', C=1))` 

In that example the svm hyperparameters of kernel and 'C' are fixed ('rbf' and 1). However, these can be optimized by testing classification with multiple parameters, with cross validation (see Selection of hyperparameters section).  For example we could use a 5 fold cross validation:

```
clf_svm_0 = make_pipeline(Vectorizer(), StandardScaler(), svm.SVC(kernel='rbf', C=1))
scores = cross_val_score(clf_svm_0, data_UN, labels_UN, cv=5)
for i in range(len(scores)):   
    print('Accuracy of ' + str(i+1) + 'th fold is ' + str(scores[i]) + '\n')
```

We could use also [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html?highlight=gridsearchcv#sklearn.model_selection.GridSearchCV) to search for the best performing parameters. We can specify which cross validation strategy we choose. In this example a stratifiedKfold is used to select the best Kernel and C from a list: 
``` 
#svm
clf_svm_pip = make_pipeline(Vectorizer(), StandardScaler(), svm.SVC(random_state=42))
parameters = {'svc__kernel':['linear', 'rbf', 'sigmoid'], 'svc__C':[0.1, 1, 10]}
gs_cv_svm = GridSearchCV(clf_svm_pip, parameters, scoring='accuracy', cv=StratifiedKFold(n_splits=5), return_train_score=True)
```

(*Note: in the pipeline function the double underscore is used to specify parameters of an element of the function: e.g., svc__kernel means it will define the parameter 'kernel' from the svc in the pipeline*) 



## Cross-validation 
Measuring prediction accuracy is central to decoding. To assess a decoder, select one in various alternatives or tune its parameters. Cross-validation is the standard tool to measure predictive power and tune parameters in decoding. 


<img src='https://user-images.githubusercontent.com/13642762/211787368-060cf4b2-f7ad-47bd-bf13-b03afdc9a7e7.png' height='200px' width='300px'>

<sub> https://scikit-learn.org/stable/modules/cross_validation.html</sub>


The following article reviews caveats and contains guidelines on the choice of cross validation methods:

Varoquaux, G. et al.,2017 Assessing and tuning brain decoders: Cross-validation, caveats, and guidelines. *NeuroImage*. https://doi.org/10.1016/J.NEUROIMAGE.2016.10.038


#### Illustration of k-fold cross-validation 

<img src='https://user-images.githubusercontent.com/13642762/211788402-e18acc1b-017e-4e7b-9fac-c5e77969c410.png' height='200px' width='300px'>

<sub> https://scikit-learn.org/stable/modules/cross_validation.html</sub>

Important concepts for CV (from Varoquax et al., 2017): 
#### Estimating predictive power
We need to measure the ability of our decoder (mapping brain images to e.g., epoch labels) to *generalize* to new data. We measure the error between the predicted label and the actual label. In CV the data is split in *training* and *test* (unseen by the model, used to compute a prediction error).
*  Training and test sets must be *independent*. E.g., in fMRI time-series we need a time separation enough to warrant independent observations

* *Sufficient test data*. To get enough power for the prediction error for each split of cross-validation. 
Because we have limited amount of data we need a *balance* between keeping enough training data for a good fit while having enough data for the test. 
Neuroimaging studies tend to use **leave-one-out** cross validation (LOOCV), i.e., leaving a single sample out at each training-test split. This gives ample data for training, maximizes test-set variance and does not yield stable estimates of predictive accuracy.It might be preferable then to instead leave out 10-20% of the data, like in 10-fold CV.  It might also be beneficial to increase the number of spllits while keeping a ration between train and test size. Thus **k-fold** can be replaced by **repeated random splits** of the data (aka repeated learning-testing or shuffleSplit). Such splits should be consistent with the dependence structure across observations, or the training set could be stratified to  avoid class imbalance. In neuroimaging good strategies often involve leaving out sessions or subjects. 

#### Selection of hyper-parameters
In standard statistics, fitting a model on abundant data can be done without choosing a meta-parameter: all model parameters can be estimated from data, e.g., with a maximum-likelihood criterion. But in high-dimensional settings the model of parameters are much larger than the sample size, we need **regularization**. 

Regularization restricts the model complexity to avoid **overfitting** the data (e.g., fitting noise, not being able to generalize).  For instance, we can use low-dimensional PCA in discriminant analysis, or select a small number of voxels with a sparse penalty. If we do *too much* regularization, the models **underfit** the data, i.e., they are too constrained by the prior and do not exploit the data enough. 

In general the best tradeoff is a data-specific choice governed by the statistical power of the prediction task: the amount of data and our signal-to-noise ratio. 
The typical **bias-variance** problem: more variance leads to overfit , but too much bias leads to underfit. 

##### *Nested-cross validation* 
How much regularization? A common approach is to use CV to measure predictive power for various choices of regularization and keep the values that maximize predictive power. With this approach the *amount of regularization* becomes a parameter to adjust on the data, thus predictive performance measured in the CV loop cannot reliably assess predictive performance. The standard procedure is then to refit the model on available data, and test predictive performance on new data: a *validation set*. 

A *nested cross-validation* repeteadly splits the data into *validation* and *decoding* sets to perform the decoding. The decoding is, in turn, done by spliting a given validation set in *training* and *test* sets. This forms an inner "nested" CV loop used to set up *regularization hyper-parameter*, while the external loop cvarying the validation set is used to measure prediction performance. 

![image](https://user-images.githubusercontent.com/13642762/207826874-76aa9fa1-3ca9-4e77-9ecb-40f5a61d1b03.png)

##### *Model averaging*

How to choose the best model in a family of good models? One option is to average the predictions of a set of models.
 A simple version of this is *bagging* using *bootstrap*: random reamplings of the data to generate many train sets and corresponding models that are then (their predictions)averaged. If the errors between the models are independent enough they will average out and the model will have less variance and better performance. This is an appealing option for neuroimaging, where linear models are often used a decoders. 

>We can use a variant of CV and model averaging: instead of selecting the hyper-parameter values that minimize the mean test error across splits, we can select *for each split* the model that minimizes the corresponding test error and *tenb* average these models across splits.

#### Model selection for neuroimaging decoders
The main challenge in neuroimaging for model-selection is the scarcity of data relative to their dimensionality.  Another aspect is that beyond predictive power, interpreting the model weights is relevant. 

##### Common decoders and their regularization

Neuroimaging studies frequently use **support vector machine** (SVM) and  **logistic regressions** (Log-Reg). Both classifiers learn a linear model by minimizing the sum of a *loss -L*(data-fit term) and a *penalty -p* ( a 'regularization energy' term that favors simpler models). The regularization parameter (*C*) controls the bias-variance tradeoff, with smaller values meaning strong regularization. 
In SVM the loss used is a *hinge* loss: flat and zero for well-classified samples and the misclassification cost increases linearly with the distance to the decision boundary. For logistic regression, it is a *logistic loss*, a soft, exponentially-decreasing version of the hinge loss. 

The most common regularization is the L<sub>2</sub> (*Ridge regression). Strong SVM-L<sub>2</sub> combined with hing loss means that SVM build their decision functions by combining a small number of training images. Similarly, in logistic regression the loss has no flat region, thus every sample is used, but some very weakly. 
The L<sub>1</sub> ( *Lasso regression*) penality, on the other hand, imposes sparsity on the weights: that is a strong regularization means that the weight maps are mostly comprised of zero voxels (in fMRI)

##### Parameter tuning 
Neuroimaging publication often do not discuss their choice of decoder hyper-parameters. Other state that they use the 'default' value (e.g., C = 1 for SVMs). Standard ML practice favors setting them by nested cross-validation. For *non-sparse* L<sub>2</sub> penalized models the amount of regularization often does not strongly influence the weight maps of the decoder 

## Classification scores
The function [skearn.metrics.classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) can build a text report showing the main classification metrics like precision, recall, f1-score and accuracy. Accuracy or other metrics can be directly obtaind using sklearn's accuracy_score() or precision_recall_fscore_support() functions. If you use precision_recall_fscore_support() with average='macro' parameter, it calculates each metric by averaging all classes without weights. 

**accuracy** is one of the most common metrics, but not enough to conclude a model is performing than another. It may be deceptive, for example when a model classifies a majority of the instances to one class, accuracy can still be high if the classes are highly imbalanced. Another case would be when false postive and false negative have different consequences (e.g., in medical domain). Precision, recall and f1-score (which combines precision and recall) are also recommended by some authors https://neuro.inf.unibe.ch/AlgorithmsNeuroscience/Tutorial_files/ApplyingMachineLearningMethods_1.html.  Also the AUC of the ROC is proposed as unbiased metric when dealing with a *two-class problem*.

#### Receiver operating characteristic (ROC)
The ROC curve is applied to the obtained classification probabilities and is summarized with the AUC. The ROC curve represents the *true-positive* rate (i.e., hits; correctly classified trials) as a function of the *false-positive* rate (i.e., false alarms, missclassified). A diagonal ROC of 50% shows chance level classification score (n hits = n false alarms). A **area under the curve (AUC)** of 100 % (upper left bound of the diagonal) is a perfect positive prediction with no false positive, perfect decoding. The AUC measure of the ROC is unbiased to imbalanced problems and independent of the statistical distribution of the classes. The AUC is thus considered a sensitive,nonparametric criterion-free measure of generalization. 

#### Precision, recall and f-measures (1-score)
- *Precision* is the ability of the classifier not to label as positive a sample that is negative. See [average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)

- *Recall* is the ability of the classifier to find all the positive samples. See [precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve)

- *F-measure ([F-1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html), F<sub>b</sub>)* are a weighted harmonic mean of the precision and recall. Best value is 1 and worst is 0. (Maths note: *harmonic mean* is the reciprocal of the arithmetic mean of the reciprocal. Reciprocal or multiplicative inverse is one of a pair of numbers that, when multiplied together equals 1. F1 = 2 * (precision * recall) / (precision + recall)

### Confusion matrix
The [sklearn.metrics.confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix) computes the confusion matrix to evaluate accuracy of a classification. The confusion matrix *C* is a matrix such that *C*<sub>i,j</sub> is equal to the number of observations known to be in group *i* and predicted to be in the group *j*. In binary classification, the count of true negatives is *C*<sub>0,0</sub>, false negatives is *C*<sub>1,0</sub>, true positives is *C*<sub>1,1</sub> and false positives is *C*<sub>0,1</sub>. 

<img src = "https://user-images.githubusercontent.com/13642762/208434436-e32d3db5-47fb-4416-afea-7a4348ab65d6.png" width="265" height = "225">

<sub>Example of a confusion matrix from the scikit-learn documentation </sub>



## Applications
Here there are several possibilities for using multivariate (e.g., all sensors) information to decode cognitive/experimental manipulations from brain activitiy. The MNE documentation shows an example of a code implementation (https://mne.tools/stable/auto_examples/decoding/decoding_time_generalization_conditions.html#) 
for the following paper on temporal generalization method: King & Dehaene, 2014 doi:10.1016/j.tics.2014.01.002. For another example in EEG/MEG see for instance Marti et al., 2015 https://doi.org/10.1016/j.neuron.2015.10.040. These example show several analyses: 

### Time-resolved MVPA
The classifier is trained at each time sample within each subject to isolate topographical patterns (i.e., information from all sensors) that can best differentiate between two conditions (if more than two classes usually referred to as *multiclass*). 

Methods from Marti et al., 2015: 
* Cross-validation: 5-fold stratified CV procedure was used for within-subject analysis. At *each time point* the MEG data was randomly split into 5 folds of trials and normalized (Z score of each channel-time feature within the cross-validation). *Stratified* means that the same proportion of each classwas kept within each fold. 

* Classification: SVM trained with a fixed penalty parameter *C* = 1 on 4 folds and the left out trials were used as test set. The SVM found the hyperplane (in this case a topography) that best separated the two classess without overfitting. A *weighting procedure* equalized the contribution of each class to the definition of the hyperplane. This procedure was iteratively applied for each time sample of each fold. 

### Generalization accross time 
Here the goal is to provide information in detail about the sequence of (neural) processing stages engaged in a particular task. 

The classifiers trained at each time are tested on their ability to discriminate conditions at all other time samples. This *temporal generalization* (King & Dehaene, 2014; see also Dehaene et al.,2016 chapter https://link.springer.com/chapter/10.1007/978-3-319-28802-4_7#Sec1) results in a matrix of training time x testing time. Each row corresponds to the time at which the classifier is trained and each column to the time at which it was tested.  Its diagonal corresponds to classifiers trained and tested on the same time sample. Training one classifier at time t and generalizing it over time t' is done within the cross-validation, so that t and t' come from independent sets of trials. 

The basic interpretation is that how a decoder trained at time t generalizes to data from another time point t' would reveal whether the neural code changes over time. This analyses can show, for example, a diagonal pattern of temporal generalization, indicating that each classifier only generalizes for a limited period of time. If each time sample is associated with a slightly different pattern of EEG/MEG activity this can be interpreted as suggesting serial recruitment of different brain areas, each for a short time. 

![image](https://user-images.githubusercontent.com/13642762/207869802-0a5f9d4e-7bc2-4e21-9068-55a544f466c4.png)

<sub> Image from Dehaene et al., 2016. (https://link.springer.com/chapter/10.1007/978-3-319-28802-4_7#Sec1)
Temporal decoding applied to an auditory violation paradigm, the local/global paradigm (from King et al. 2013a). (a) Experimental design: sequences of five sounds sometimes end with a different sound, generating a local mismatch response. Furthermore, the entire sequence is repeated and occasionally violated, generating a global novelty response (associated with a P3b component of the event-related potential). (b, c) Results using temporal decoding. A decoder for the local effect (b) is trained to discriminate whether the fifth sound is repeated or different. This is reflected in a diagonal pattern, suggesting the propagation of error signals through a hierarchy of distinct brain areas. Below-chance generalization (in blue) indicates that the spatial pattern observed at time t tends to reverse at time t′. A decoder for the global effect (c) is trained to discriminate whether the global sequence is frequent or rare. This is reflected primarily in a square pattern, indicating a stable neural pattern that extends to the next trial. In all graphs, t = 0 marks the onset of the fifth sound</sub>


### Generalization accross conditions
Following temporal generalization. Here the goal is to see how different processing stages may change between experimental conditions (e.g., slowed, speeded, deleted, inserted 'stages'). A classifier is trained in a condition and tested on its ability to generalize to another. The resulting temporal generalization matrix may then indicate how information processing changed. A series of classifiers can be trained to discriminate conditions in certain type of trials and are then applied to different type of trials. 

![image](https://user-images.githubusercontent.com/13642762/207885018-cfe53290-8b94-45ae-86e3-38129eea53e1.png)

<sub>Example figure from King et al., 2014, https://doi.org/10.1016/j.tics.2014.01.002</sub>


## ML Glossary

This is a machine learning glossary in the context of multivariate pattern analysis, copied from a paper on describing a matlab toolbox for MVPA: [https://www.frontiersin.org/articles/10.3389/fnins.2020.00289/fullMVPA ](https://www.frontiersin.org/articles/10.3389/fnins.2020.00289/full).

For an in-depth introduction to machine learning refer to standard textbooks (Bishop, 2007; Hastie et al., 2009; James et al., 2013).

* **Binary classifier**. A classifier trained on data that contains two classes, such as in the “faces vs. houses” experiment. If there is more than two classes, the classifier is called a multi-class classifier.

* **Classification**. One of the primary applications of MVPA. In classification, a classifier takes a multivariate pattern of brain activity (referred to as feature vector) as input and maps it onto a categorical brain state or experimental condition (referred to as class label). In the “faces vs. houses” experiment, the classifier is used to investigate whether patterns of brain activity can discriminate between faces and houses.

* **Classifier**. An algorithm that performs classification, for instance Linear Discriminant Analysis (LDA) and Support Vector Machine (SVM).

* **Classifier output**. If a classifier receives a pattern of brain activity (feature vector) as input, its output is a predicted class label e.g., “face.” Many classifiers are also able to produce class probabilities (representing the probability that a brain pattern belongs to a specific class) or decision values.

* **Class label**. Categorical variable that represents a label for each sample/trial. In the “faces vs. houses” experiment, the class labels are “face” and “house.” Class labels are often encoded by numbers, e.g., “face” = 1 and “house” = 2, and arranged as a vector. For instance, the class label vector [1, 2, 1] indicates that a subject viewed a face in trial 1, a house in trial 2, and another face in trial 3.

* **Cross-validation**. To obtain a realistic estimate of classification or regression performance and control for overfitting, a model should be tested on an independent dataset that has not been used for training. In most neuroimaging experiments, there is only one dataset with a restricted number of trials. K-fold cross-validation makes efficient use of such data by splitting it into k different folds. In every iteration, one of the k folds is held out and used as test set, whereas all other folds are used for training. This is repeated until every fold served as test set once. Since cross-validation itself is stochastic due to the random assignment of samples to folds, it can be useful to repeat the cross-validation several times and average the results. See Lemm et al. (2011) and Varoquaux et al. (2017) for a discussion of cross-validation and potential pitfalls.

* **Data**. From the perspective of a classifier or regression model, a dataset is a collection of samples (e.g., trials in an experiment). Each sample consists of a brain pattern and a corresponding class label or response. In formal notation, each sample consists of a pair (x, y) where x is a feature vector and y is the corresponding class label or response.

* **Decision boundary**. Classifiers partition feature space into separate regions. Each region is assigned to a specific class. Classifiers make predictions for a test sample by looking up into which region it falls. The boundary between regions is known as decision boundary. For linear classifiers, the decision boundary is also known as a hyperplane.

* **Decision value**. Classifiers such as LDA and SVM produce decision values which can be thresholded to produce class labels. For linear classifiers and kernel classifiers, a decision value represents the distance to the decision boundary. The further away a test sample is from the decision boundary, the more confident the classifier is about it belonging to a particular class. Decision values are unitless.

* **Decoder**. An alternative term for a classifier or regression model that is popular in the neuroimaging literature. The term nicely captures the fact that it tries to invert the encoding process. In encoding e.g., a sensory experience such as viewing a face is translated into a pattern of brain activity. In decoding, one starts from a pattern of brain activity and tries to infer whether it was caused by a face or a house stimulus.

* **Feature**. A feature is a variable that is part of the input to a model. If the dataset is tabular with rows representing samples, it typically corresponds to one of the columns. In the “faces vs. houses” experiment, each voxel represents a feature.

* **Feature space**. Usually a real vector space that contains the feature vectors. The dimensionality of the feature space is equal to the number of features.

* **Feature vector**. For each sample, features are stored in a vector. For example, consider a EEG measurement with three electrodes Fz, Cz, and Oz and corresponding voltages 40, 65, and 97 μV. The voltage at each EEG sensor represents a feature, so the corresponding feature vector is the vector [40, 65, 97] ∈ ℝ3.

* **Fitting (a model)**. Same as training.

* **Hyperparameter**. A parameter of a model that needs to be specified by the user, such as the type and amount of regularization applied, the type of kernel, and the kernel width γ for Gaussian kernels. From the user's perspective, hyperparameters can be nuisance parameters: it is sometimes not clear a priori how to set them, but their exact value can have a substantial effect on the performance of the model.

* **Hyperparameter tuning**. If it is unclear how a hyperparameter should be set, multiple candidate values can be tested. Typically, this is done via nested cross-validation: the training set is again split into separate folds. A model is trained for each of the candidate values and its performance is evaluated on the held-out fold, called validation set. Only the model with the best performance is then taken forward to the test set.

* **Hyperplane**. For linear classifiers, the decision boundary is a hyperplane. In the special case of a two-dimensional feature space, a hyperplane corresponds to a straight line. In three dimensions, it corresponds to a plane.

* **Loss function**. A function that is used for training. The model parameters are optimized such that the loss function attains a minimum value. For instance, in Linear Regression the sum of squares of the residuals serves as a loss function.

* **Metric**. A quantitative measure of the performance of a model on a test set. For example, precision/recall for classification or mean squared error for regression.

* **Model**. In the context of this paper, a model is a classifier or regression model.

* **Multi-class classifier**. A classifier trained on data that contains three or more classes. For instance, assume that in the “faces vs. houses” experiment additional images have been presented depicting “animals” and “tools.” This would define four classes in total, hence classification would require a multi-class classifier.

* **Overfitting**. Occurs when a model over-adapts to the training data. As a consequence, it will perform well on the training set but badly on the test set. Generally speaking, overfitting is more likely to occur if the number of features is larger than the number of samples, and more likely for complex non-linear models than for linear models. Regularization can serve as an antidote to overfitting.

* **Parameters**. Models are governed by parameters e.g., beta coefficients in Linear Regression or the weight vector w and bias b in a linear classifier.

* **Regression**. One of the primary applications of MVPA (together with classification). Regression is very similar to classification, but it aims to predict a continuous variable rather than a class label. For instance, in the ‘faces vs. houses' experiment, assume that the reaction time of the button press has been recorded, too. To investigate the question “Does the pattern of brain activity in each trial predict reaction time?,” regression can be performed using reaction time as responses.

* **Regression model**. An algorithm that performs regression, for instance Ridge Regression and Support Vector Regression (SVR).

* **Regularization**. A set of techniques that aim to reduce overfitting. Regularization is often directly incorporated into training by adding a penalty term to the loss function. For instance, L1 and L2 penalty terms are popular regularization techniques. They reduce overfitting by preventing coefficients from taking on too large values.

* **Response**. In regression, responses act as the target values that a model tries to predict. They play the same role that class labels play in classification. Unlike class labels, responses are continuous e.g., reaction time.

* **Searchlight analysis**. In neuroimaging analysis, a question such as “Does brain activity differentiate between faces and houses?” is usually less interesting than the question “Which brain regions differentiate between faces and houses?.” In other words, the goal of MVPA is to establish the presence of an effect and localize it in space or time. Searchlight analysis intends to marry statistical sensitivity with localizability. It is a well-established technique in the fMRI literature, where a searchlight is defined e.g., as a sphere of 1 cm radius, centered on a voxel in the brain (Kriegeskorte et al., 2006). All voxels within the radius serve as features for a classification or regression analysis. The result of the analysis is assigned to the central voxel. If the analysis is repeated for all voxel positions, the resultant 3D map of classification accuracies can be overlayed on a brain image. Brain regions that have discriminative information then light up as peaks in the map. Searchlight analysis is not limited to spatial coordinates. The same idea can be applied to other dimensions such as time points and frequencies.

* **Testing**. The process of applying a trained model to the test set. The performance of the model can then be quantified using a metric.

* **Test set**. Part of the data designated for testing. Like with training sets, test sets are automatically defined in cross-validation, or they can arise naturally in multi-site studies or in experiments with different phases.

* **Training**. The process of optimizing the parameters of a model using a training set.

* **Training set**. Part of the data designated for training. In cross-validation, a dataset is automatically split into training and test sets. In other cases, a training set may arise naturally. For instance, in experiments with different phases (e.g., memory encoding and memory retrieval) one phase may serve as training set and the other phase as test set. Another example is multi-site studies, where a model can be trained on data from one site and tested on data from another site.

* **Underfitting**. Occurs when a classifier or regression model is too simple to explain the data. For example, imagine a dataset wherein the optimal decision boundary is a circle, with samples of class 1 being inside the circle and samples of class 2 outside. A linear classifier is not able to represent a circular decision boundary, hence it will be unable to adequately solve the task. Underfitting can be checked by fitting a complex model (e.g., kernel SVM) to data. If the complex model performs much better than a more simple linear model (e.g., LDA) then it is likely that the simple model underfits the data. In most neuroimaging datasets, overfitting is more of a concern than underfitting.
