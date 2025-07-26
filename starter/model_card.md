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


Overall, the model achieves a Precision of 0.73, Recall of 0.61 and F1‑score of 0.66, but slice analysis shows significant variation: for example, recall for race_Amer-Indian-Eskimo drops to 0.375 while it exceeds 0.85 for education_Doctorate.

Mid‑level education groups (e.g. HS‑grad) show F1 below 0.50, whereas advanced degrees score above 0.80, indicating strong proxy bias via education.

Gender slices reveal higher precision for males (0.76) but lower recall (0.53) compared to females (0.62), suggesting unequal error rates across sex.


## Caveats and Recommendations
- **Generalization:** Performance may degrade on data distributions different from training.  
- **Monitoring:** Evaluate metrics and fairness regularly in production.  
- **Retraining:** Update the model periodically to mitigate concept drift.
