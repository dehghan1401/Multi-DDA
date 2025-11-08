### Multi-DDA: Drug-Disease Association Prediction using a Hybrid Graph Convolutional Network with Multi-modal Drug Representations

 <p align="justify">  The overall schematic of the Multi-DDA approach is given in Figure . As it is shown, the proposed model is composed of the graph convolutional layer (GCL), graph attention layer (GAL), and fully connected layer (FCL). The recent work fed the heterogeneous graph of drug diseases into the graph convolutional network. To recap, there are some challenges in the recent works, like ignoring the raw sequence of drug compounds and utilizing the provided drug-disease association in the input dataset as an initial representation of the nodes. To cope with these challenges, in the proposed model, not only the constructed heterogeneous graph is used as input, but also the auxiliary knowledge about the drugs, including enzyme, target, pathway, and substructure representation of the drugs, is fed as input to provide more discriminative information to update the drug nodes' features. </p>
<img width="948" height="1254" alt="image" src="https://github.com/user-attachments/assets/918dec1f-2bf9-4256-86e9-2f5403795485" />


## Datasets

    data/drug_dis.csv is the drug_disease association matrix, which contains 18416 associations between 269 drugs and 598 diseases.
    data/drug_sim.csv is the drug similarity matrix of 269 diseases, which is calculated based on drug target features.
    data/dis_sim.csv is the disease similarity matrix of 598 diseases, which is calculated based on disease mesh descriptors.



## Requirements

    tensorflow>=2.8.0
    numpy>=1.21.0
    scipy>=1.7.0
    scikit-learn>=1.0.0
    pandas>=1.3.0

## Data Setup
    
    Place your data files in the ../data/ directory:
    drug_sim.csv: Drug similarity matrix
    dis_sim.csv: Disease similarity matrix
    drug_dis.csv: Drug-disease association matrix
    drug_names.csv: Drug names (optional)
    disease_names.csv: Disease names (optional)
    Ensure the SCMFDD dataset is available at:/content/drive/My Drive/DrugDisease/SCMFDD/SCMFDD_Dataset.mat
