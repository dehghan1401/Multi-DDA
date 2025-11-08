Multi-DDA: Drug-Disease Association Prediction using a Hybrid Graph Convolutional Network with Multi-modal Drug Representations
 

Datasets

    data/drug_dis.csv is the drug_disease association matrix, which contains 18416 associations between 269 drugs and 598 diseases.
    data/drug_sim.csv is the drug similarity matrix of 269 diseases, which is calculated based on drug target features.
    data/dis_sim.csv is the disease similarity matrix of 598 diseases, which is calculated based on disease mesh descriptors.



Requirements

    tensorflow>=2.8.0
    numpy>=1.21.0
    scipy>=1.7.0
    scikit-learn>=1.0.0
    pandas>=1.3.0

Data Setup
    
    Place your data files in the ../data/ directory:
    drug_sim.csv: Drug similarity matrix
    dis_sim.csv: Disease similarity matrix
    drug_dis.csv: Drug-disease association matrix
    drug_names.csv: Drug names (optional)
    disease_names.csv: Disease names (optional)
    Ensure the SCMFDD dataset is available at:/content/drive/My Drive/DrugDisease/SCMFDD/SCMFDD_Dataset.mat
