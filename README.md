# Deep Learning Project
Jordan Axelrod, Rathi Kashi, Rajas Pandey

## Directory Structure
### Data
This folder contains the FashionMnist dataset 
### Docs
This contains all the final generated images and the intermediate loss function tables
### Literature
This folder contains all of the papers referenced for the project
### src
####Noise Regularization: ipynb file consisting of:
1. Dataloaders for the noise regularization model 
2. Noise Regularization model architecture and training
3. Proposed Model architecture and training
4. Metrics and Visualisations 
####WSGAN: Scripts for implementing the GAN model consisting of:
1. TrainStep1.py: GAN generator using contrastive learning
2. TrainingStep2.py: Learning decoder through decoder classification
3. TrainStep3.py: Finetune classification using local and global classifier architecture
4. All other files are classes and helper functions
5. modeldump: folder contains checkpoints at different stages
####DataProc: make_data.py creates the dataloaders for the GAN model
