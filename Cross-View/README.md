# Cross_view_actionRecognition
This project provides cross-view, cross-subject experiments on N-UCLA dataset


Note: for cross-view experiments, using 'setup1' to do ablation studies, all pre-trained models are based on 'setup1' configuration; 
      setup1: view1,view2 training, view3 testing

                    
PROJECT STRUCTURE : 
1) data: lists

2) dataset : dataloader

3) modelZoo: model implementations

4) pretrained: provided pretrained DYAN, note that, 'xxx.pth', is trained with MSE(GT, dictionary * sparsecode), while
            'xxx_BI.pth', is trained with reconstruction loss MSE(GT, dictionary * (sparsecode * binarycode))
	    
5) trainClassifier_CV.py (training script, CV)

6) testClassifier_CV.py. (testing script, CV)

---------------------------------------------------------------------------------------------------------------------------------------------------------

TO START, creating a folder named 'pretrained', go to : 
https://drive.google.com/drive/folders/1LNDdPdKDSFWhTtY0XOOEKbjKxktbIJTp?usp=sharing
Download pretrained model, 


---------------------------------------------------------------------------------------------------------------------------------------------------------

USAGES:
1. Software requiremtns: python >=3.6, pytorch>=1.13

2. Important configuration hyper-paramters: 
   1) clip --> 'Single': each sequence has ONE sampled input clip
               'Multi': each sequence has MORE than one sampled input clips

   2) dataType--> '2D' : using 2D skeleton data
                  '3D' : using 3D skeleton data

   3) Alpha, lam1, lam2 : weights for each loss term, where
                          Alpha --> binary loss
                          lam1 --> classification loss
                          lam2 --> mse loss
   4) FistaLam: hyper-parameters for Fista of DyanEncoder

   5) Dyan pretrain: pretrained DYAN encoder on the dataset, following architectures load pre-trained paramethers in advance

3. network architectures:

   1) classificationWSparseCode() : classification with Sparse code,
      
   2) Fullclassification() : dynamic stream(DIR), classification with binary code,
           
   3) twoStreamClassification(): dynamic + context streams(DIR + CIR), 
           
   4) RGBAction(): context stream (CIR)

4. Experiment parameters:
   
   - REGULAR DYAN (baseline)	
   	1) classification + sparse code 
   	
   		optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=lr, weight_decay=0.001, momentum=0.9)
		
   		- 'Single':
   		
			a. Alpha : lam1: lam2 = 0 : 2 : 1
	
        		b.  fistaLam = 0.1
	
        		c.  lr = 1e-4, 

   	2) Dynamic Stream (DIR) 
		
		optimizer = torch.optim.SGD([{'params':filter(lambda x: x.requires_grad, net.sparseCoding.parameters()), 'lr':lr},
                              {'params':filter(lambda x: x.requires_grad, net.Classifier.parameters()), 'lr':lr_2}], weight_decay=0.001, momentum=0.9)
   		
		- 'Single':
		
   			a. Alpha : lam1 : lam2 = 0.1 : 1 : 0.5
	
			c.  fistaLam = 0.1
        
			d.  lr = 5e-4, lr_2 = 1e-4, (lr:sparsecode, lr_2:classifier)
		
		- 'Multi':
	
	
   - Re-weighted DYAN(RH-dyan)
      will do;

   - 2-stream 
      will do; 


