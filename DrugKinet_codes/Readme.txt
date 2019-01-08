DrugKinet codes folder:

Scrape.py: web scraping code that extracts data from DurgInet website

readKinases.m: Code that reads KinaseMats folder and extract the bioactivity matrix

readMat.py: Loads required data into workspace

baseAlgos.py: Runs basic feature based methods like Logistic regression, SVMs and enroll networks. Need to make changes in the code in order to run different algorithms.

collabFilt.py: contains low rank matrix factorization codes

getSmiles_and_ecfp4.py: extracts SMILES descriptors and ecfps for all the compounds using remit and pubchempy packages in python