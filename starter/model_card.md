# Model Card

UDACITY excersice. Model intention is to predict income from selected attributes.

## Model Details
- **Type:** `RandomForestClassifier` (scikit‑learn), _n_estimators_=100, _max_depth_=None, _random_state_=42, _n_jobs_=-1 :contentReference[oaicite:1]{index=1}  
- **Inputs:** 8 categorical features (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`) plus continuous attributes :contentReference[oaicite:2]{index=2}  
- **Preprocessing:** OneHotEncoder for categorical features; LabelBinarizer for the income label :contentReference[oaicite:3]{index=3}  

## Intended Use
- **Primary:** Predict whether an individual’s annual income exceeds \$50K using the UCI Census Income dataset.  
- **Audience:** Data scientists and analysts building risk‑ or segmentation‑based models.  
- **Limitations:** Not intended for high‑stakes decisions without further validation.

## Training Data
- **Source:** UCI “Census Income” dataset, cleaned by stripping whitespace and periods. :contentReference[oaicite:4]{index=4}  
- **Split:** 80% train / 20% test (using `train_test_split(random_state=42)`). :contentReference[oaicite:5]{index=5}

## Evaluation Data
- **Hold‑out set:** 20% test split, processed with the same encoders as training.  
- **Approach:** Single train/test split; no cross‑validation.

## Metrics
- **Precision:** 0.73  
- **Recall:** 0.61  
- **F1‑score:** 0.66  

## Ethical Considerations
- **Bias risk:** Under‑represented demographic groups may be misclassified.  
- **Sensitive features:** `race`, `sex`, `native-country` :contentReference[oaicite:6]{index=6}

## Caveats and Recommendations
- **Generalization:** Performance may degrade on data distributions different from training.  
- **Monitoring:** Evaluate metrics and fairness regularly in production.  
- **Retraining:** Update the model periodically to mitigate concept drift.
