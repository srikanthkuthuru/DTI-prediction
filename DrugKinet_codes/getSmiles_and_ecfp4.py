'''
Author: Srikanth Kuthuru, Rice University
1. This code calculates the ecfp4 fingerprints for all the unique compounds extracted from drugkinet database.
'''

#Need pubchempy and RDkit packages
#RDkit requires a separate virtual environment
import pandas as pd
import pubchempy as pcp
df = pd.read_csv('PubChemIDs.csv')
a = df.PID
a[674] = '9549295';
a[165] = '125697';
smiles = ['']*len(a)
for i in range(5):#range(len(a)):
    pid = int(a[i])
    try:
        c = pcp.Compound.from_cid(pid)
        smiles[i] = c.canonical_smiles
    except:
        print('CID not present')
            
    #print(i, smiles[i])
    print(i, c.synonyms[0])
    
s = pd.Series(smiles, name = 'Smiles')
df['Smiles'] = s
#%%

fps = ['']*len(s) #Circular Fingerprints
from rdkit import Chem
from rdkit.Chem import AllChem
for i in range(len(s)):
    try:
        mol = Chem.MolFromSmiles(s[i])
        fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
        fps[i] = fp.ToBitString()
    except:
        print('Smiles not present')
    print(i)
         
s = pd.Series(fps, name = 'ecfp4')
df['ecfp4'] = s
#df.to_csv('ecfp4.csv')



