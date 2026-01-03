## Overview
This project aims to use Machine Learning Techniques to predict Cardiovascular Diseases by the use of commonly collected laboratory and clinical data. It ia an early stage Cardiovascular risk
screening model.

## Dataset
This dataset comes from anonymized clinical files and it includes a set of patient informations and lab work such as Fasting Blood Sugar, Creatinine, Cholesterol e.t.c.

## Models Used
- Logistic Regression
- Random Forest Classifier

## Methodological Decision
The initial modeling [Heart Risk Model A] got a linear regression accuracy of 0.8731707317073171 and a random forest accuracy of 1.0. This high accuracy of the random forest was first thought
to be because of high data leakage that are functionally diagnostic example Vessels colored by flouroscopy and thalassemia. and the reaon for the moderate linear regression score is because it
is constrained.
After exclusion of both Vessels colored by flouroscopy and thalassemia, the Random forest accuracy still remained 1.0 and so the following debugging sequence where followed.
- Verify column identity: this was done to know which columns excactly are being used or seen by the random forest model and to make sure columns like Target, Vessels colored by flouroscopy and
thalassemia were exculuded.
- Check for cantamination (Train - Test contamination): this was done to know if the model is being tested on data that has already been memorized. This often happens when data manipulation is done without
resetting the index.
- Confirm no target leakage through encoded categories: this was done because sometimes, leakage hides in plain site. A column like Chest pain type might include a ategory called "Typical Angina"
which is a good indicator for Cardiovascular disease.
- Constraining the Random Forest Classifier: Random Forest default classifiers are too permisive. On small datasets they create a rule for every single patient, this is over fitting. And by limiting
the size of the leaf and depth, the model is forced to look for broad patterns like high cholesterol usually indicates high risk and not just specific individuals.

## Files
- Heart_risk_model A.py
- Heart_risk_model B.py : Main training and evaluation script
- Heart_data.csv : Clinical dataset
- Summary Results Model A
- Summary Results Model B
- README
- Heart_logistic_model.pkl : Trained logistic regression model
- Heart_random_forest_model.pkl : Trained random forest model

## Key Findings
After the debugging was done, the Random Forest Model generated an acurracy of [0.8097560975609757] and the Logistics Regression gave and acuracy of [0.7804878048780488]. This
accuracy result shows that diagnosis leakage have been removed, no memorization. It also showes genuine generalization under uncertinty and appropriate seperation between linear and non-linear
modeling capacity.
This behaviour now matches real life clinical reality.  

## Authour
Abdulazeez Hanif

## Notes
Trained model files (.pkl) were generated locally when the script is executed and are not stored in the repository.