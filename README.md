Excerpt of code analysing data to develop automated classifier
This code will not run as certain code has been removed
A lot of the data loading / cleaning / preprocessing is removed

Background:

Had ~3 weeks to read material, data, then make models / code
Whole 3 Month project (I am not joining) intends to create classifier, and explain it
They had:
  1. engineered features from company,
  2. Also had a deep neural network for classification from previous project.
I was interested in determining whether a deep neural network was needed,
or if it simply inhibits the follow-up task of explaining the classifier.
I did not pursue attempts to optimise a given model - this was more a scoping project

There were 2 datasets:
  1. Dataset 1 had only their engineered features and classifications.
  2. Dataset 2 was split across multiple excel / csv files and was partially labelled.

They had transformed their engineered features via scaling but not documented how too clearly, difficult to replicate.
The results between Dataset 1 and Dataset 2 using same features were very different.
This indicates need to follow up for further clarity on how their transformation was done.

Conclusion:
  1. If they can replicate their scaling and get similar performance as they did for Dataset 1:
     They should simply keep their features, as they are meaningful and performant.
  2. If they cannot be replicated, proposed model: mlp_I3B4Y1, is a recommended candidate:
     It uses their features (with my attempt at their scaling) combined with quantile-based data.
     The features are therefore interpretable. The model is a simple MLP that should be further optimised.
     It had an accuracy of ~93% versus the project's DNN's existing 85.8%.*
     *Care with these metrics: I created my own data pipeline, so tests are not like-for-like.
  3. If manually selected features are not desired, proposed model: mlp_I2AY1, is good starting point
     It had an accuracy of ~92%. I think this can be easily increased with additional tuning.
     I also think errors in the autoencoder should be fed in as additional inputs, but out of time!
