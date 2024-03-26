# Lab2_Simpsons-Characters-Recognition

Goal
-
The main purpose of this experiment is to establish an image classifier for predicting and classifying processed images of characters from the following 50 episodes of The Simpsons series. Moreover, the competition is on Kaggle (https://www.kaggle.com/competitions/machine-learning-2023nycu-classification/overview).
- abraham_grampa_simpson (labeled as 0)
- agnes_skinner (labeled as 1)
- apu_nahasapeemapetilon (labeled as 2)
- barney_gumble (labeled as 3)
- bart_simpson (labeled as 4)
- brandine_spuckler (labeled as 5)
- carl_carlson (labeled as 6)
- charles_montgomery_burns (labeled as 7)
- chief_wiggum (labeled as 8)
- cletus_spuckler (labeled as 9)
- comic_book_guy (labeled as 10)
- disco_stu (labeled as 11)
- dolph_starbeam (labeled as 12)
- duff_man (labeled as 13)
- edna_krabappel (labeled as 14)
- fat_tony (labeled as 15)
- gary_chalmers (labeled as 16)
- gil (labeled as 17)
- groundskeeper_willie (labeled as 18)
- homer_simpson (labeled as 19)
- jimbo_jones (labeled as 20)
- kearney_zzyzwicz (labeled as 21)
- kent_brockman (labeled as 22)
- krusty_the_clown (labeled as 23)
- lenny_leonard (labeled as 24)
- lionel_hutz (labeled as 25)
- lisa_simpson (labeled as 26)
- lunchlady_doris (labeled as 27)
- maggie_simpson (labeled as 28)
- marge_simpson (labeled as 29)
- martin_prince (labeled as 30)
- mayor_quimby (labeled as 31)
- milhouse_van_houten (labeled as 32)
- miss_hoover (labeled as 33)
- moe_szyslak (labeled as 34)
- ned_flanders (labeled as 35)
- nelson_muntz (labeled as 36)
- otto_mann (labeled as 37)
- patty_bouvier (labeled as 38)
- principal_skinner (labeled as 39)
- professor_john_frink (labeled as 40)
- rainier_wolfcastle (labeled as 41)
- ralph_wiggum (labeled as 42)
- selma_bouvier (labeled as 43)
- sideshow_bob (labeled as 44)
- sideshow_mel (labeled as 45)
- snake_jailbird (labeled as 46)
- timothy_lovejoy (labeled as 47)
- troy_mcclure (labeled as 48)
- waylon_smithers (labeled as 49)

File Discription
-
1. code
  - trainer.py : Main code for training and testing.
  - trainer.ipynb : Main code on Colab for training and testing.
  - data_augmentation.py : Code for data augmentation.
2. Result
  - record.txt : The record of training processing.
  - test.csv : Predictions based on testing dataset.

  ------------------------------------------------------------
  Due to the file size limitations on GitHub, you can access the following files via the provided cloud link: https://drive.google.com/drive/folders/1rIPBwiTL2EOaz568ogO7nscxxgwEGf-C?usp=drive_link
  
  - Best_Model.pth : Saved model based on the best validation accuracy.
  -  Model.pth : The lastest saved model.
  -  loss_acc.pth : The file saved training loss, validation loss, training accuracy and validation accuracy in each epoch.
  -  Learning curve (loss).png : The visualization figure of learning curve based on training and validation loss.
  -  Learning curve (acc).png : The visualization figure of learning curve based on training and validation accuracy.
  -  Confusion matrix.png : The visualization figure of confusion matrix for validation.
  

How to start up
-

Please open file trainer.py. And in the following block, def parse_config():, you can set the path of the folders for training data and testing data respectively. Moreover, you can modify the values of hyperparameters and the path of output files by your own. After setting these parameters, you can run this code to start training.

        ###########################################################################
        ##                          Argparse Setting                             ##
        ###########################################################################
        def parse_config():
            parser = argparse.ArgumentParser()
            parser.add_argument("--train_data_path", default='D:/Master/2023ML/Lab2/Data/train_agu_2', help="1st-level folder of train data")
            parser.add_argument("--train_overview", default='D:/Master/2023ML/Lab2/train_data_agu_2.csv', help="path of overview file for training data")
            parser.add_argument("--test_data_path", default='D:/Master/2023ML/Lab2/Data/test-final', help="folder of testing data")

            parser.add_argument("--n_epochs_1", type=int, default=15, help="number of epochs of first-stage training")
            parser.add_argument("--n_epochs_2", type=int, default=30, help="number of epochs of second-stage training")
            parser.add_argument("--num_class", type=int, default=50, help="number of categories for classification")

            parser.add_argument("--batch_size", type=int, default=200, help="size of the batches")
            parser.add_argument("--num_workers", type=int, default=4, help="num_workers for Dataloader")
            parser.add_argument("--lr", type=float, default=0.0035, help="adam: learning rate")
            parser.add_argument("--opt",  default='Adam', help="Optimizer:'SGD','RMSprop','Adagrad','Adam'")
            parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
            parser.add_argument("--b2", type=float, default=0.97, help="adam: decay of second order momentum of gradient")
            parser.add_argument("--loss", default='Cross_entropy', help="Loss func:'MAE','MSE','NLL(Negative Log-Likelihood)','Cross_entropy'")

            parser.add_argument('--outf', default='D:/Master/2023ML/Lab2/Result', help='folder to output record and model')
            parser.add_argument('--log', default='D:/Master/2023ML/Lab2/Resultrecord_2.txt', help='path to record')
            parser.add_argument('--font', default=20, help='font size for output figure')
            parser.add_argument('--resume', default='', help='path to resume model weight')
            args = parser.parse_args(args=[])
            return args

Methodology
-
![image](https://github.com/Machine-Learning-NYCU/the-simpsons-characters-recognition-challenge-iii-310581003/assets/145360863/7c681cba-9383-46bf-8f06-f284ab21c5a4)

1. Data pre-processing

     In order to make accurate predictions on testing data that has already undergone exaggerated image processing, we must first perform data augmentation on the training data. In this experiment, data augmentation is applied by performing five image processing operations on each training image. Therefore, the effective training data quantity becomes six times the original, and the following lines show the data augmentation process.

    | Training data : 465259 images.
  
    | Validation data : 116315 images.
  
    | Testing data : 10791 images.
   
    - RandomApply
        - Code
  
              trans_1 = T.Compose([
   
                      T.ToTensor(),  # Convert PIL image to tensor
                      T.RandomApply([T.RandomHorizontalFlip()], p=0.3),
                      T.RandomApply([T.RandomVerticalFlip()], p=0.3),
                      T.RandomApply([T.RandomRotation(10)], p=0.3),

                      T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.3),
                      T.RandomGrayscale(p=0.3),
                      T.RandomInvert(p=0.3),
                      T.RandomPosterize(bits=2, p=0.3),
                      T.RandomApply([T.RandomSolarize(threshold=1.0)], p=0.3),
                      T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2)], p=0.3),

                      T.RandomApply([AddGaussianNoise(0., 0.05)], p=0.2),  # mean and std
                      T.RandomApply([AddPoissonNoise(lam=0.1)], p=0.3),  # mean and std
                      T.RandomApply([AddSpeckleNoise(noise_level=0.1)], p=0.3),
                      T.RandomApply([AddSaltPepperNoise(salt_prob=0.05, pepper_prob=0.05)], p=0.3),

                      T.RandomApply([T.RandomPerspective(distortion_scale=0.6, p=1.0)], p=0.3),
                      T.RandomApply([T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))], p=0.3),
                      T.RandomApply([T.ElasticTransform(alpha=250.0)], p=0.3),

                      T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.3),
                      T.RandomApply([AddGaussianNoise(0., 0.001)], p=1.0),  # mean and std
                      T.ToPILImage()  # Convert tensor back to PIL image for saving

            ])

        - Example

          ![image](https://github.com/Machine-Learning-NYCU/the-simpsons-characters-recognition-challenge-iii-310581003/assets/145360863/94a17210-62ad-4c87-b164-e2e72e2ca96b)          

    - RandomChoice & RandomApply
        - code

              trans_set_aff = [ 
                      T.RandomPerspective(distortion_scale=0.6, p=1.0),
                      T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
              ]
              trans_set_noise= [ 
                      AddGaussianNoise(0., 0.05),
                      AddGaussianNoise(0., 0.001),
                      AddPoissonNoise(lam=0.1),
                      AddSpeckleNoise(noise_level=0.1),
                      AddSaltPepperNoise(salt_prob=0.05, pepper_prob=0.05)
              ]
              trans_set_flip = [ 
                      T.RandomHorizontalFlip(),
                      T.RandomVerticalFlip(),
              ]
              trans_set_PGISA = [ 
                      T.RandomPosterize(bits=2, p=0.1),
                      T.RandomGrayscale(p=0.1),
                      T.RandomInvert(p=0.1),
                      T.RandomSolarize(threshold=1.0),
                      T.RandomAdjustSharpness(sharpness_factor=2)
              ]
              trans_3 = T.Compose([
                      T.ToTensor(),
                      T.RandomChoice(trans_set_flip),
                      T.RandomApply([T.RandomRotation(10)], p=0.3),
                      T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.3),
                      T.RandomChoice(trans_set_PGISA),
                      T.RandomChoice(trans_set_noise),
                      T.RandomApply([T.RandomPerspective(distortion_scale=0.6, p=1.0)], p=0.3),
                      T.RandomApply([T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))], p=0.3),
                      T.RandomApply([T.ElasticTransform(alpha=250.0)], p=0.3),

                      T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.3),
                      T.RandomApply([AddGaussianNoise(0., 0.001)], p=1.0),  # mean and std
                      T.ToPILImage() 
              ])

        - Example

          
          ![image](https://github.com/Machine-Learning-NYCU/the-simpsons-characters-recognition-challenge-iii-310581003/assets/145360863/43e2a84d-d128-4695-a865-58572341ae44)

2. Transfer learning

   - Pre-trained Model
     
      In this experiment, Transfer Learning is used to train a classifier for the Simpsons image dataset. The pre-trained model utilized for this purpose is the EfficientNet_V2_S_Weights.IMAGENET1K_V1. The reason for using this model as a pre-trained model is its relatively low parameter count of 21.5 million and its high level of accuracy. For more information on this model, you can refer to the following URL [https://pytorch.org/vision/stable/models.html)https://pytorch.org/vision/stable/models.html].

   - Feature extraction & Fine-tuning

     This section primarily aims to apply the pre-trained model to a customizable dataset. This section is divided into two stages:
     
        - Feature Extraction: In this stage, the weights of the earlier layers of the pre-trained model are retained, and the rest layers are trained using the training dataset from this experiment.

        - Fine-tuning: In the second stage, fine-tuning is performed by using the training dataset from this experiment to adjust the weights of the entire model.

3. Hyperparameters :
    - Epoch :
        - First stage (Feature extraction) : 15
        - Second stage (Fine-tuning) : 85
    - Optimizer : Adam
    - Learning rate : 0.0035
    - Loss function : Cross entropy
    - Batch size : 200

Code Discriptions for trainer.py
-

1. Argparse Settings

    Initial variable setting including the path to input & output files, hyperparameter settings.

        ###########################################################################
        ##                          Argparse Setting                             ##
        ###########################################################################
        def parse_config():
            parser = argparse.ArgumentParser()
            parser.add_argument("--train_data_path", default='D:/Master/2023ML/Lab2/Data/train_agu_2', help="1st-level folder of train data")
            parser.add_argument("--train_overview", default='D:/Master/2023ML/Lab2/train_data_agu_2.csv', help="path of overview file for training data")
            parser.add_argument("--test_data_path", default='D:/Master/2023ML/Lab2/Data/test-final', help="folder of testing data")

            parser.add_argument("--n_epochs_1", type=int, default=15, help="number of epochs of first-stage training")
            parser.add_argument("--n_epochs_2", type=int, default=30, help="number of epochs of second-stage training")
            parser.add_argument("--num_class", type=int, default=50, help="number of categories for classification")

            parser.add_argument("--batch_size", type=int, default=200, help="size of the batches")
            parser.add_argument("--num_workers", type=int, default=4, help="num_workers for Dataloader")
            parser.add_argument("--lr", type=float, default=0.0035, help="adam: learning rate")
            parser.add_argument("--opt",  default='Adam', help="Optimizer:'SGD','RMSprop','Adagrad','Adam'")
            parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
            parser.add_argument("--b2", type=float, default=0.97, help="adam: decay of second order momentum of gradient")
            parser.add_argument("--loss", default='Cross_entropy', help="Loss func:'MAE','MSE','NLL(Negative Log-Likelihood)','Cross_entropy'")

            parser.add_argument('--outf', default='D:/Master/2023ML/Lab2/Result', help='folder to output record and model')
            parser.add_argument('--log', default='D:/Master/2023ML/Lab2/Resultrecord_2.txt', help='path to record')
            parser.add_argument('--font', default=20, help='font size for output figure')
            parser.add_argument('--resume', default='', help='path to resume model weight')
            args = parser.parse_args(args=[])
            return args

2. Category mapping :
   Mapping between labels and Simpsons characters.

       ###########################################################################
       ##                            Categories                                ##
       ###########################################################################
        Categories = {0  : "abraham_grampa_simpson",
                      1  : "agnes_skinner",
                      2  : "apu_nahasapeemapetilon",
                      3  : "barney_gumble",
                      4  : "bart_simpson",
                      5  : "brandine_spuckler",
                      6  : "carl_carlson",
                      7  : "charles_montgomery_burns",
                      8  : "chief_wiggum",
                      9  : "cletus_spuckler",
                      10 : "comic_book_guy",
                      11 : "disco_stu",
                      12 : "dolph_starbeam",
                      13 : "duff_man",
                      14 : "edna_krabappel",
                      15 : "fat_tony",
                      16 : "gary_chalmers",
                      17 : "gil",
                      18 : "groundskeeper_willie",
                      19 : "homer_simpson",
                      20 : "jimbo_jones",
                      21 : "kearney_zzyzwicz",
                      22 : "kent_brockman",
                      23 : "krusty_the_clown",
                      24 : "lenny_leonard",
                      25 : "lionel_hutz",
                      26 : "lisa_simpson",
                      27 : "lunchlady_doris",
                      28 : "maggie_simpson",
                      29 : "marge_simpson",
                      30 : "martin_prince",
                      31 : "mayor_quimby",
                      32 : "milhouse_van_houten",
                      33 : "miss_hoover",
                      34 : "moe_szyslak",
                      35 : "ned_flanders",
                      36 : "nelson_muntz",
                      37 : "otto_mann",
                      38 : "patty_bouvier",
                      39 : "principal_skinner",
                      40 : "professor_john_frink",
                      41 : "rainier_wolfcastle",
                      42 : "ralph_wiggum",
                      43 : "selma_bouvier",
                      44 : "sideshow_bob",
                      45 : "sideshow_mel",
                      46 : "snake_jailbird",
                      47 : "timothy_lovejoy",
                      48 : "troy_mcclure",
                      49 : "waylon_smithers"}

3. Build the training dataset

   Sketch the training data and build the training datasets.

         ###########################################################################
        ##                              Preparing Data                          ##
        ###########################################################################

        #---------------------------- Dataloader ---------------------------#
        class Simpsons_DataSet(Dataset):
            def __init__(self, mode):

                args = parse_config()

                self.img_path = args.train_data_path
                self.img_names = np.squeeze(pd.read_csv(args.train_overview, usecols=["name"]).values)
                self.labels    = np.squeeze(pd.read_csv(args.train_overview, usecols=["label"]).values)
                assert len(self.img_names)==len(self.labels), 'Error : Data and labels length not the same'
                self.data_len=len(self.img_names)

                # Define the image augmentation transformations
                self.transformations = T.Compose([
                        #T.ToPILImage(),  # Convert tensor back to PIL image for saving
                        #T.AugMix(severity= 6,mixture_width=2),
                        #T.RandomPosterize(bits=2, p=0.1),
                        T.ToTensor(),

                        #T.RandomApply([T.RandomHorizontalFlip()], p=0.1),
                        #T.RandomApply([T.RandomVerticalFlip()], p=0.1),
                        #T.RandomApply([T.RandomRotation(10)], p=0.1),

                        #T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.1),
                        #T.RandomGrayscale(p=0.1),
                        #T.RandomInvert(p=0.1),

                        #T.RandomApply([T.RandomSolarize(threshold=1.0)], p=0.05),
                        #T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2)], p=0.1),

                        #T.RandomApply([AddGaussianNoise(0., 0.05)], p=0.1),  # mean and std
                        #T.RandomApply([AddPoissonNoise(lam=0.1)], p=0.1),  # mean and std
                        #T.RandomApply([AddSpeckleNoise(noise_level=0.1)], p=0.1),
                        #T.RandomApply([AddSaltPepperNoise(salt_prob=0.05, pepper_prob=0.05)], p=0.1),

                        #T.RandomApply([T.RandomPerspective(distortion_scale=0.6, p=1.0)], p=0.1),
                        #T.RandomApply([T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))], p=0.1),
                        #T.RandomApply([T.ElasticTransform(alpha=250.0)], p=0.1),

                        #T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.1),

                        #T.RandomApply([AddGaussianNoise(0., 0.001)], p=1.0),  # mean and std

                        T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

                 ])


               print(f'>> Found {self.data_len} images for training ...')
               with open(args.log, 'a') as f:
                   f.write(f'>> Found {self.data_len} images for training ...\n')
                   f.write(f'\n')
            def __len__(self):
                return self.data_len

            def __getitem__(self, index):

                single_img_name=os.path.join(self.img_path,self.img_names[index]+'.jpg')

                single_img=Image.open(single_img_name).convert("RGB")
                single_img = T.Resize((224, 224))(single_img)
                img=self.transformations(single_img)
                label=self.labels[index]
                #print(f'labels : {label}')

                return img, label

4. Definition of model 
   
    Definitions of pre-trained model (EfficientNet) and the number of neurons in the last layer.
   
       ###########################################################################
        ##                                   Model                              ##
        ###########################################################################

        class ResNet50(nn.Module):
            def __init__(self,num_class,pretrained=False):
            """
            Args:
                num_class: #target class
                pretrained:
                      True: the model will have pretrained weights, and only the last layer's 'requires_grad' is True(trainable)
                     False: random initialize weights, and all layer's 'require_grad' is True
            """
                super(ResNet50,self).__init__()
                self.model=models.efficientnet_v2_s(pretrained=pretrained)
                if pretrained:
                  for param in self.model.parameters():
                    param.requires_grad=False

                num_neurons = self.model.classifier[1].in_features
                #num_neurons=self.model.fc.in_features
                #num_neurons = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(num_neurons, num_class, bias=True)
                #self.model.fc=nn.Linear(num_neurons, num_class, bias=True)

            def forward(self,X):
                out=self.model(X)
                return out

5. Training & Validation

    Definitions of training loop and validation.
   
        ###########################################################################
        ##                         Train & Validation                           ##
        ###########################################################################
        def train(model, train_loader, valid_loader, epochs, loss_func, optimizer, device, state):
        """
          Args:
            model: resnet model
            train_loader: training dataloader
            valid_loader: validation dataloader
            loss_func: loss function
            optimizer: optimizer
            epochs: number of training epoch
            device: gpu/cpu
            state: the state of the last training (used for resuming)
          Returns:
            df_loss: with column 'epoch','loss_train','loss_valid'
            df_acc: with column 'epoch','acc_train','acc_valid'
            best_model_wts: the best accuracy saved model
        """
            args = parse_config()
            num_class = args.num_class
            df_loss=pd.DataFrame()
            df_acc=pd.DataFrame()
            df_loss['epoch']=range(1,epochs+1)
            df_acc['epoch']=range(1,epochs+1)
            best_model_wts=None

            model.to(device)

            if args.resume != '':

              checkpoint = torch.load(os.path.join(args.resume, 'Model.pth'))
              model.load_state_dict(checkpoint['model_state_dict'])
              optimizer.load_state_dict(checkpoint['optim_state_dict'])
              #saved_param = torch.load(os.path.join(args.resume, 'param.pth'))
              best_evaluated_acc = checkpoint['best_acc']
              start_epoch =checkpoint['last_epoch']+1
              loss_train=checkpoint['loss_train']
              loss_valid=checkpoint['loss_valid']
              acc_train=checkpoint['acc_train']
              acc_valid=checkpoint['acc_valid']


            else:
              start_epoch = 1
              best_evaluated_acc = 0
              loss_train=list()
              loss_valid=list()
              acc_train=list()
              acc_valid=list()


          with open(args.log, 'a') as f:
              f.write(f'>>> Start training... \n')

          for epoch in range(start_epoch,epochs+1):
            """
              train
            """
              with torch.set_grad_enabled(True):
                  model.train()
                  t_loss=0
                  correct=0
                  for i, (img, label) in enumerate(tqdm(train_loader, desc='Epoch '+str(epoch))):

                      img, label=img.to(device),label.to(device,dtype=torch.long)
                      predict=model(img)
                      loss=loss_func(predict,label)
                      t_loss+=loss.item()
                      correct+=predict.max(dim=1)[1].eq(label).sum().item()
                    """
                      update
                    """
                      optimizer.zero_grad()
                      loss.backward()  # bp
                      optimizer.step()
                  t_loss/=len(train_loader.dataset)
                  t_acc=100.*correct/len(train_loader.dataset)
                  loss_train.append(t_loss)
                  acc_train.append(t_acc)

            """
              evaluate
            """
              _, v_loss, v_acc=evaluate(model, valid_loader, loss_func, device)
              loss_valid.append(v_loss)
              acc_valid.append(v_acc)
              print(f'epoch{epoch:>2d} | Train loss:{t_loss:.4f} | Train acc:{t_acc:.2f} | Valid loss:{v_loss:.4f} | Valid acc:{v_acc:.2f}\n')
              with open(args.log, 'a') as f:
                  f.write(f'epoch{epoch:>2d} | Train loss:{t_loss:.4f} | Train acc:{t_acc:.2f} | Valid loss:{v_loss:.4f} | Valid acc:{v_acc:.2f}\n')

              # update best_model_wts
              if v_acc > best_evaluated_acc:
                  best_evaluated_acc=v_acc
                  best_model_wts=copy.deepcopy(model)
                  torch.save(model, os.path.join(args.outf, 'Best_Model.pth'))

              checkpoint_dict = {'last_epoch': epoch,
                                 'model_state_dict': model.state_dict(),
                                  'optim_state_dict': optimizer.state_dict(),
                                 'best_acc' : best_evaluated_acc,
                                 'loss_train' : loss_train,
                                 'loss_valid' : loss_valid,
                                 'acc_train' : acc_train,
                                 'acc_valid' : acc_valid,
                                 'state' : state}
               torch.save( checkpoint_dict, os.path.join(args.outf, 'Model.pth'))


              df_loss['loss_train']=loss_train
              df_loss['loss_valid']=loss_valid
              df_acc['acc_train']=acc_train
              df_acc['acc_valid']=acc_valid

              return df_loss, df_acc, best_model_wts

       def evaluate(model, valid_loader, loss_func, device):
        """
          Args:
            model: resnet model
            valid_loader: validation dataloader
            device: gpu/cpu
          Returns:
            confusion_matrix: (num_class,num_class) ndarray
            total_loss : validation average loss
            acc: validation accuracy
        """

            args = parse_config()
            num_class = args.num_class
            confusion_matrix=np.zeros((num_class,num_class))

            with torch.set_grad_enabled(False):
                model.eval()
                correct=0
                total_loss=0
                for img, label in valid_loader:
                    img,label=img.to(device),label.to(device,dtype=torch.long)
                    predict=model(img)
                    predict_class=predict.max(dim=1)[1]
                    loss=loss_func(predict,label)
                    total_loss+=loss.item()
                    correct+=predict_class.eq(label).sum().item()
                    for i in range(len(label)):
                        confusion_matrix[int(label[i])][int(predict_class[i])]+=1
                total_loss/=len(valid_loader.dataset)
                acc=100.*correct/len(valid_loader.dataset)

            # normalize confusion_matrix
            confusion_matrix=confusion_matrix/confusion_matrix.sum(axis=1).reshape(num_class,1)

            return confusion_matrix, total_loss,acc

6. Plotting
   
     To plot the learning curve and confusion matrix for model assessment and return figures for saving.

       ###########################################################################
        ##                               Plot                                   ##
        ###########################################################################
        def plot(dataframe1, mode):
        """
          Arguments:
              dataframe1: dataframe with 'epoch','loss_train','loss_valid','acc_train','acc_valid' columns of without pretrained weights model
              title: figure's title
          Returns:
              figure: an figure
        """
            fig=plt.figure(figsize=(10,6))
            for name in dataframe1.columns[1:]:
            plt.plot(range(1,1+len(dataframe1)),name,data=dataframe1,label=name)

            plt.xlabel('Epochs')
            if mode == 'loss' :
                plt.ylabel('Loss')
                plt.title('Learning curve (loss)')
            elif mode == 'acc':
                plt.ylabel('Accuracy(%)')
                plt.title('Learning curve (acc)')
            plt.legend()
            return fig

        def plot_confusion_matrix(confusion_matrix):
            args = parse_config()
            label_name = ["abraham_grampa_simpson", "agnes_skinner", "apu_nahasapeemapetilon", "barney_gumble", "bart_simpson",
                          "brandine_spuckler", "carl_carlson", "charles_montgomery_burns", "chief_wiggum", "cletus_spuckler",
                          "comic_book_guy", "disco_stu", "dolph_starbeam", "duff_man", "edna_krabappel",
                          "fat_tony", "gary_chalmers", "gil", "groundskeeper_willie", "homer_simpson",
                          "jimbo_jones", "kearney_zzyzwicz", "kent_brockman", "krusty_the_clown", "lenny_leonard",
                          "lionel_hutz", "lisa_simpson", "lunchlady_doris", "maggie_simpson", "marge_simpson",
                          "martin_prince", "mayor_quimby", "milhouse_van_houten", "miss_hoover", "moe_szyslak",
                          "ned_flanders", "nelson_muntz", "otto_mann", "patty_bouvier", "principal_skinner",
                          "professor_john_frink", "rainier_wolfcastle", "ralph_wiggum", "selma_bouvier", "sideshow_bob",
                          "sideshow_mel", "snake_jailbird", "timothy_lovejoy", "troy_mcclure", "waylon_smithers"]

            fig, ax = plt.subplots(figsize=(25,25))
            ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
            ax.xaxis.set_label_position('top')

            for i in range(confusion_matrix.shape[0]):
                for j in range(confusion_matrix.shape[1]):
                    ax.text(i, j, '{:.2f}'.format(confusion_matrix[j, i]), va='center', ha='center')
            ax.set(xticks=range(0, 50, 1), xticklabels=label_name, yticks=range(0, 50,1), yticklabels=label_name)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

            ax.set_xlabel('Predicted label', fontsize = 15)
            ax.set_ylabel('True label', fontsize = 15)
            return fig

7. Testing
   
   Inference the Simpsons characters for the testing data based on trained model and output the test.csv file.

       ###########################################################################
        ##                                   Test                               ##
        ###########################################################################
        def test(model, outf):
        """
          Args:
            model: trained model
            testing_dataset: testing dataset
            outf: path to output file
        """
            args = parse_config()
            df_id =list()
            df_character = list()
            img_path = args.test_data_path
            # Define the image augmentation transformations
            transformations = T.Compose([
                T.ToTensor(),  # Convert tensor back to PIL image for saving
                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
            ])
            img_names =list()
            count = 0
            with torch.set_grad_enabled(False):
                model.eval()
                correct=0
                i = 1
                for filename in os.listdir(img_path):
                    count=count+1
                    file_name=Path(filename).stem
                    img_names.append(file_name)
                    single_img_name=os.path.join(img_path,file_name+'.jpg')
                    single_img=Image.open(single_img_name).convert("RGB")
                    single_img = T.Resize((224, 224))(single_img)
                    img=transformations(single_img)
                    img=img.to(device)
                    predict=model(img.unsqueeze(0))
                    df_id.append(file_name)
                    print(torch.argmax(predict,1))
                    df_character.append(Categories.get(int(torch.argmax(predict,1))))
                    i=i+1

            data_len = count
            df = pd.DataFrame()

            df['id'] = df_id
            df['character'] = df_character
            df.to_csv(os.path.join(outf, 'test.csv'))

            return data_len, df

Results & Disscusion
-

1. Results

   - Best validation :
       - Accurcy : 96.76 %
       - Loss : 0.0006
         
   - Best testing on Kaggle :
       - Private score : 0.98451
    
   - Learning curve :
       - training loss versus validation loss

            ![Learning curve (loss)](https://github.com/Machine-Learning-NYCU/the-simpsons-characters-recognition-challenge-iii-310581003/assets/145360863/a1ca89fa-693b-4d4c-be83-560d49248c6a)

      - training accuracy versus validation accuracy

          ![Learning curve (acc)](https://github.com/Machine-Learning-NYCU/the-simpsons-characters-recognition-challenge-iii-310581003/assets/145360863/3d850857-487c-4eb0-a483-58698a7ec219)

    - Confusion matrix for best validation

        ![Confusion matrix](https://github.com/Machine-Learning-NYCU/the-simpsons-characters-recognition-challenge-iii-310581003/assets/145360863/6247d2e6-4d2a-45bf-b528-6143ea88954c)

  2. Disscussion

     - Feature maps

         The image below displays the feature maps obtained after passing one of the testing images through the first layer of the model. It is noticeable that some of the feature maps are focused on extracting the edges from the image.

       ![layer_0](https://github.com/Machine-Learning-NYCU/the-simpsons-characters-recognition-challenge-iii-310581003/assets/145360863/1b48f519-51a3-4132-acba-6cc4cfc1be5d)

     - Compare to the results of applying different settings

          In fact, at the very beginning of this experiment, training was performed on a dataset that had undergone only one round of data augmentation. The pre-trained model used was ResNet50. The initial parameter settings are as follows. During the training process, the highest validation accuracy achieved was 70.19%. However, when predictions for testing data were uploaded to Kaggle, the private score received was only 0.01985. Consequently, this experiment later adopted more rounds of data augmentation and switched to using EfficientNet as the pre-trained model to strive for better predictive results.

         - Pre-trained model : ResNet50
         - Epoch :
             - First stage : 10
             - Second stage : 10

        - Optimizer : Adam
        - Learning rate : 0.025
        - Loss function : Cross entropy
        - Batch size : 24
