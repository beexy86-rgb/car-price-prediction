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

# ulazni podaci 

Ulazni podaci se nalaze u folderu data, u fajlu cars.csv. Podaci su dalje obrađeni kroz čišćenje i dodavanje novih karakteristika, a ti novi fajlovi su takođe snimljeni u folder data.

# pokretanje koda

U projektu postoji više fajlova - neki od njih korisnik koristi direktno, odnosno pokreće ih, a neki fajlovi služe kao ispomoć fajlovima koje korisnik pokreće.
Da bi se kod pokrenuo počev ispočetka, od sirovih podataka, potrebno je prvo pokrenuti skriptu: 

main.py, 

koja se nalazi u korenom folderu. Nakon izvršavanja koda dobiće se fajl sa očišćenim podacima i fajl sa dodatnim karakteristikama, oba snimljena u folder data.

Zatim se pokreće skripta:

model_training.py, 

koja se nalazi u folderu src. Na ovaj način se trenira model linearne regresije i čuva u folderu models.

Pokretanjem skripte:

make_prediction.py

u korenom folderu dobijaju se predikcije modela na osnovu slučajno odabranih redova iz test skupa podataka. Prikazuju se stvarne i prognozirane cene automobila.

Pokretanjem skripte:

model_evaluation.py

u folderu src dobijaju se metrike za vrednovanje treniranog modela linearne regresije.

Pokretanjem skripte:

model_comparison.py 

u folderu src radi se poređenje različitih regresionih modela: Linear Regression, Decision Tree, Random Forest i Gradient Boosting. Korišćene metrike su: mae, mse, rmse i R2. Na osnovu izlaznih rezultata bira se najbolji model.

Linearna regesija se pokazala kao najlošiji model. Ostala tri modela imaju slične metrike, ali je Random Forest najbolji.


# odabir najboljeg modela

Od četiri trenirana modela najbolje se pokazao Random Forest (pogledati tabelu ispod). Ovaj model u proseku najmanje greši (prosečno greši u proceni cene automobila za 1054 dolara). I kada se gleda ponderisana greška, koja strže kažnjava velike promašaje, Random Forest model je najbolji. Koefiicijent determinacije ovog modela takođe najbolje objašnjava varijacije ciljne promenljive. 

 model          mae           mse         rmse        r2
2      Random Forest  1054.495236  6.833551e+06  2614.106157  0.895777
1      Decision Tree  1353.645909  1.059164e+07  3254.480125  0.838459
3  Gradient Boosting  1551.845505  9.819563e+06  3133.618227  0.850235
0  Linear Regression  2092.432919  1.943184e+07  4408.155618  0.70363
