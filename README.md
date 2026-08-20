# car-price-prediction
Predikcija cene automobila na osnovu njegovih karakteristika

# struktura projekta

car-price-prediction/
├── data/
│   └── cars.csv
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── model_comparison.py
├── models/
│   └── car_price_model.joblib
├── README.md
└── requirements.txt

# odabir najboljeg modela

Od četiri trenirana modela najbolje se pokazao Random Forest. Ovaj model u proseku najmanje greši (prosečno greši u proceni cene automobila za 1054 dolara). I kada se gleda ponderisana greška, koja strže kažnjava velike promašaje, Random Forest model je najbolji. Koefiicijent determinacije ovog modela takođe najbolje objašnjava varijacije ciljne promenljive. 

 model          mae           mse         rmse        r2
2      Random Forest  1054.495236  6.833551e+06  2614.106157  0.895777
1      Decision Tree  1353.645909  1.059164e+07  3254.480125  0.838459
3  Gradient Boosting  1551.845505  9.819563e+06  3133.618227  0.850235
0  Linear Regression  2092.432919  1.943184e+07  4408.155618  0.70363
