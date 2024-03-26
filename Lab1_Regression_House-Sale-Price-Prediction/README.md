# Lab1_Regression_House-Sale-Price-Prediction
Goal
-
The goal of this project is to predict the house prices based on the following features

>> The date house was sold (year/month/day), Number of bedrooms/house, Number of bathrooms/bedrooms, Square footage of the home, Square footage of the lot, Total floors (levels) in house, House which has a view to a waterfront, The times has been viewed, How good the condition is ( Overall ), Overall grade given to the housing unit, Square footage of house apart from basement, Square footage of the basement, Built Year, Year when house was renovated, Zip code, Latitude coordinate & Longitude coordinate, Living room area, Lotsize area.

Files Discriptions
-
1. Data :

   >> 。 metadata.csv : supplemental information about the data
   >>
   >> 。 train-v3.csv : raw training dataset
   >>
   >> 。 valid-v3.csv : raw validation dataset
   >>
   >> 。 test-v3.csv : raw testing dataset
   >>
   >> 。 train_data_add_norm.csv : normalized training dataset based on z-score normalization
   >>
   >> 。 valid_data_add_norm.csv : normalized validation dataset based on z-score normalization
   >>
   >> 。 test_data_add_norm.csv : normalized testing dataset based on z-score normalization

2. Feature_Engineering.py :
   >> do z-score normalization for raw datasets

3. train.py :
   >> Main code including initial variables definition, dataset building, model definition, training, validation, testing, learning curve plotting, etc.

4. Result :
   >> 。 test_prediction_best.csv : inferences for testing data
   >> 
   >> 。 Learning_curve_best.png : learning curve plot
   >> 
   >> 。 record_best.txt : including hyperparameter settings, training loss and validation loss for each epochs, best validation loss
   >>
   >> 。 Saved_Model_best.pth : trained model

Methodology
-
1. Data dicriptions

   1. Number of input features : 21
   2. Input features :

   
   3. Number of training data : 12967
   4. Number of training data : 2161
   5. Number of training data : 6485
   
1. Architecture
   
   Regression model based on multilayer perceptron (MLP). The protocol of architecture can be seen as following,

   ![image](https://github.com/Machine-Learning-NYCU/regression-house-sale-price-prediction-challenge-310581003/assets/145360863/c67ad5aa-4bad-4b4a-91c9-247c83077737)

   
4. Hyperparameters
   1. Number of epoch :  800
   2. Learning rate : 0.25
   3. Batch size : 5500
   4. Optimizer : Adam
   5. Loss function : MAE

How to start up
-
Please open file train.py. And in the following block, def parse_config():, you can set the path of the input files for train_data, valid_data and test_data respectively. Moreover, you can modify the values of hyperparameters and the path of output files by your own.

      def parse_config():
        parser = argparse.ArgumentParser()
        parser.add_argument("--train_data", default='D:/Master/2023ML/Lab1/train_data_add_norm.csv', help="path of training data file")
        parser.add_argument("--valid_data", default='D:/Master/2023ML/Lab1/valid_data_add_norm.csv', help="path of validation data file")
        parser.add_argument("--test_data", default='D:/Master/2023ML/Lab1/test_data_add_norm.csv', help="path of testinging data file")
   
        parser.add_argument("--feature_sel", default='all', help="input feature selection")
        parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
        parser.add_argument("--batch_size", type=int, default=5500, help="size of the batches")
        parser.add_argument("--lr", type=float, default=0.05, help="adam: learning rate")
        parser.add_argument("--opt",  default='Adam', help="Optimizer:'SGD','RMSprop','Adagrad','Adam'")
        parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.97, help="adam: decay of second order momentum of gradient")
        parser.add_argument("--loss", default='MAE', help="Loss func:'MAE','MSE','NLL(Negative Log-Likelihood)','Cross_entropy'")
    
        parser.add_argument('--outf', default='D:/Master/2023ML/Lab1/Result/', help='folder to output record and model')
        parser.add_argument('--log', default='D:/Master/2023ML/Lab1/Result/record.txt', help='path to record')
        parser.add_argument('--font', default=20, help='font size for output figure')
        parser.add_argument('--save_model', default='D:/Master/2023ML/Lab1/Result/Saved_Model', help='path to resume model weight')
        args = parser.parse_args(args=[])
        return args 


Code Discriptions for train.py
-
1. Argparse Settings
   
   Initial variable setting including the path to input & output files, hyperparameter settings. 

       def parse_config():
        parser = argparse.ArgumentParser()
        parser.add_argument("--train_data", default='D:/Master/2023ML/Lab1/train_data_add_norm.csv', help="path of training data file")
        parser.add_argument("--valid_data", default='D:/Master/2023ML/Lab1/valid_data_add_norm.csv', help="path of validation data file")
        parser.add_argument("--test_data", default='D:/Master/2023ML/Lab1/test_data_add_norm.csv', help="path of testinging data file")
   
        parser.add_argument("--feature_sel", default='all', help="input feature selection")
        parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
        parser.add_argument("--batch_size", type=int, default=5500, help="size of the batches")
        parser.add_argument("--lr", type=float, default=0.05, help="adam: learning rate")
        parser.add_argument("--opt",  default='Adam', help="Optimizer:'SGD','RMSprop','Adagrad','Adam'")
        parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.97, help="adam: decay of second order momentum of gradient")
        parser.add_argument("--loss", default='MAE', help="Loss func:'MAE','MSE','NLL(Negative Log-Likelihood)','Cross_entropy'")
    
        parser.add_argument('--outf', default='D:/Master/2023ML/Lab1/Result/', help='folder to output record and model')
        parser.add_argument('--log', default='D:/Master/2023ML/Lab1/Result/record.txt', help='path to record')
        parser.add_argument('--font', default=20, help='font size for output figure')
        parser.add_argument('--save_model', default='D:/Master/2023ML/Lab1/Result/Saved_Model', help='path to resume model weight')
        args = parser.parse_args(args=[])
        return args 
2. Model Definition

   Definitions of MLP layers, activation functions.

        class Model(nn.Module):
            def __init__(self, num_input):
                super(Model, self).__init__()
                self.num_input = num_input
                self.layer = nn.Sequential(                       
                                    nn.Linear(self.num_input, self.num_input*2, bias=True),
                                    nn.ELU(),
                                    nn.Linear(self.num_input*2, self.num_input*2, bias=True),
                                    nn.ELU(),  
                                    nn.Linear(self.num_input*2, self.num_input*2, bias=True),
                                    nn.ELU(),
                                    nn.Linear(self.num_input*2 , self.num_input*2, bias=True),
                                    nn.ELU(),
                                    nn.Linear(self.num_input*2 , self.num_input*2, bias=True),
                                    nn.ELU(),
                                    nn.Linear(self.num_input*2 , self.num_input, bias=True),
                                    nn.ELU(),
                                    nn.Linear(self.num_input , 1),
                               )
        
            def forward(self, x):
                x = self.layer(x)
                return x

3. Build the dataset

   Sketch the data and build the datasets.

          def Sketch_Data (mode, feature_sel) :    
                args = parse_config()    
          ## read data for different modes
                if mode == 'train':
                    data = pd.read_csv(args.train_data, usecols=feature_sel).values        
                    labels=pd.read_csv(args.train_data, usecols=["price"]).values        
                elif mode == 'valid':
                    data = pd.read_csv(args.valid_data, usecols=feature_sel).values
                    labels=pd.read_csv(args.valid_data, usecols=["price"]).values        
                elif mode == 'test':
                    data = pd.read_csv(args.test_data, usecols=feature_sel).values
                    labels={}       
                print (f'Data:\n{data}')
                print (f'Label:\n{labels}')
    
          ## data length calculation
                data_len=len(data)
                print(f'>> Found {data_len} data for {mode}...')
                with open(args.log, 'a') as f:
                    f.write(f'>> Found {data_len} PPG segments for {mode}...\n')
                    f.write(f'\n')
                return data, labels

4. Training & Validation

   Definitions of training loop and validation.

        def train(model,loader_train,loader_valid,Loss,optimizer,epochs,device):
        """
            Args:
                model: resnet model
                loader_train: training dataloader
                loader_valid: validation dataloader
                Loss: loss function
                optimizer: optimizer
                epochs: number of training epoch
                device: gpu/cpu
            Returns:
                df_loss: with column 'epoch','loss_train','loss_test'
                 best_model_wts : the trained model with the best loss 
        """
            args = parse_config()
            df_acc=pd.DataFrame()
            df_loss=pd.DataFrame()
            df_acc['epoch']=range(1,epochs+1)
            df_loss['epoch']=range(1,epochs+1)
            best_model_wts=None
            best_evaluated_loss=10000
    
            model.to(device)
            acc_train=list()
            acc_valid=list()
            loss_train=list()
            loss_valid=list()
            with open(args.log, 'a') as f:
                f.write(f'----------------------------------------------------------\n')
                f.write(f'-                        Training                        -\n')
                f.write(f'----------------------------------------------------------\n')
            for epoch in range(1,epochs+1):
            """
                train
            """
                with torch.set_grad_enabled(True):
                model.train()
                train_total_loss=0
                correct=0
                for data,labels in  (loader_train):
                    data,labels=data.to(device),labels.to(device)
                    predict=model(data.float())
                    loss=Loss(predict,labels.float())
                    train_total_loss+=loss.item()
                    correct+=predict.max(dim=1)[1].eq(labels).sum().float().item()
                    """
                        update
                    """
                    optimizer.zero_grad()
                    loss.backward()  # bp
                    optimizer.step()
                train_total_loss/=len(loader_train.dataset)
                loss_train.append(train_total_loss)            
                """
                    validation
                """        
                acc, valid_loss=evaluate(model,loader_valid,Loss,device)
                loss_valid.append(valid_loss)
                print(f'epoch{epoch:>2d} | Training loss:{train_total_loss:.4f} | Validation loss:{valid_loss:.4f}')
                with open(args.log, 'a') as f:
                    f.write(f'epoch{epoch:>2d} | Training loss:{train_total_loss:.4f} | Validation loss:{valid_loss:.4f}\n')
                # update best_model_wts
                if valid_loss<best_evaluated_loss:
                    best_evaluated_loss=valid_loss
                    best_model_wts=copy.deepcopy(model)

                df_loss['loss_train']=loss_train
                df_loss['lossc_valid']=loss_valid
                print(f'The best valid loss : {best_evaluated_loss}')
                with open(args.log, 'a') as f:
                    f.write(f'>>>>>>> Best valid loss : {best_evaluated_loss}\n')
                  
                # save model
                torch.save(best_model_wts,os.path.join(args.save_model+'.pt'))
      
                return df_loss, best_model_wts

        def evaluate(model,loader_valid,Loss,device):
        """
            Args:
                model: resnet model
                loader_test: testing dataloader
                device: gpu/cpu
            Returns:
                confusion_matrix: (num_class,num_class) ndarray
                acc: validation accuracy
                total_loss : validation loss
        """
    
            with torch.set_grad_enabled(False):
                model.eval()
                correct=0
                total_loss=0
                for data,labels in  (loader_valid):  
                    data,labels=data.to(device),labels.to(device)
                    predict=model(data.float())
                    loss=Loss(predict,labels.float())
                    total_loss+=loss.item()
                    predict_class=predict.max(dim=1)[1]
                    correct+=predict_class.eq(labels).sum().float().item()
                    total_loss/=len(loader_valid.dataset)        
                    acc=100.*correct/len(loader_valid.dataset) 
        
                return acc, total_loss

6. Plot the learning curve

   Plot the learning curve for model assessment and return the figure for saving to 'Learning_curve.png'.
 
           def plot(df_loss, title):
            """
                Arguments:
                    df_loss: dataframe with 'epoch','loss_train','loss_test' columns of trained weights model  
                    title: figure's title
                Returns:
                    fig_loss : figure of the learning curve
            """
                fig_loss=plt.figure(figsize=(10,6))
                for name in df_loss.columns[1:]:
                    plt.plot(range(1,1+len(df_loss)),name,data=df_loss,label=name[4:])
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title(title)
                plt.legend()
                return fig_loss

7. Testing

   Inference the house prices for the testing data based on trained model and output the test.csv file.

       def test(model,testing_dataset,outf):
         """
          Args:
              model: trained model
              testing_dataset: testing dataset
              outf: path to output file
         """
          df_id =list()
          df_price = list()
          with torch.set_grad_enabled(False):
              model.eval()
              correct=0
              i = 1
              for data in testing_dataset:  
                  data=data.to(device)
                  predict=model(data.float())
                  df_id.append(i)
                  df_price.append(predict.item())
                  i=i+1 
          df = pd.DataFrame()
          df['id'] = df_id
          df['price'] = df_price
          df.to_csv(os.path.join(outf, 'test.csv')) 

Results
-
。 Best average validation loss : 29.51764771807034

。 Learning curve : 

![Learning_curve_best](https://github.com/Machine-Learning-NYCU/regression-house-sale-price-prediction-challenge-310581003/assets/145360863/f8c5570f-0a4d-45dc-be73-ced674684513)


Discussion & Conclusion
-

1. The result on Kaggle based on prediction of testing data :

   。 The score on Kaggle : 69,151.35243 (public) / 67,193.36030 (private)

   。 Comments :

     The average loss (sum of loss/len(data)) based on testing data is less than validation loss. This result might indicate that the trained model is not much general. The future work we can do is to increase the training datasets.

2. Improvements :

   。 Data normalization :
   
      To improve the performance of the model, I tried to do the data normalization. The comparison between the before and after can be seen as following. It can previosly tell that data normalization can improve much performance.

    >> 。 Before data normalization : the best validation loss = 62.947637089310504
    >> 
    >> 。 After data normalization : the best validation loss = 29.51764771807034
   
   。 Feature engineering :
   
      I also tried to improve the performance through feature engineering. The followings are the features I created based on those origin data. However, the results show that it do not improve the performance; instead, it leads to worse performance.

      1. The features created based on the origin data :

      >> 。 'cust_time_type' : the group of K-mean clustering based on feature 'sale_yr', 'sale_month', 'sale_day', 'yr_built' and 'yr_renovated'.
      >> 
      >> 。 'cust_layout_type' : the group of K-mean clustering based on feature 'bedrooms', 'bathrooms', 'floors' and 'waterfront'.
      >>
      >> 。 'cust_area_type' : the group of K-mean clustering based on feature 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15' and 'sqft_lot15'.
      >>
      >> 。 'cust_address_type' :  the group of K-mean clustering based on feature 'zipcode', 'lat' and 'long'.
      >>
      >> 。 'cust_condition_type' : the group of K-mean clustering based on feature 'condition' and 'grade'.
      >>
      >> 。 'sale_long' = (2023-'sale_yr')*365 + (12-'sale_month')*30 + (31-'sale_day')
      >>
      >> 。 'more_than_1_floor' : whether the floor of the house is more than 1 based on feature 'floors'.
      >>
      >> 。 'grade_more_than_8' : whether the grade of the house is more than 1 based on feature 'grade'.
      >>
      >> 。 'basement_or_not' : whether the basement of the house exists based on feature 'sqft_basement'.
      >>
      >> 。 'renovate_or_not' : whether the house has been renovated based on feature 'yr_renovated'.
      >>
      >> 。 'built_long' = 2023-'yr_built'.

               time_feature = ['sale_yr', 'sale_month', 'sale_day', 'yr_built', 'yr_renovated']
               layout_feature = ['bedrooms', 'bathrooms', 'floors', 'waterfront' ]
               area_feature = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']
               address_feature = ['zipcode', 'lat', 'long']
               condition_feature = ['condition', 'grade']
               #-------------------------------------------------------------------------------
               kmeans_t = KMeans(n_clusters=3)
               kmeans_t.fit(df[time_feature])
               df['cust_time_type'] = kmeans_t.predict(df[time_feature])
               #-------------------------------------------------------------------------------
               kmeans_l = KMeans(n_clusters=3)
               kmeans_l.fit(df[layout_feature])
               df['cust_layout_type'] = kmeans_l.predict(df[layout_feature])
               #-------------------------------------------------------------------------------
               kmeans_ar = KMeans(n_clusters=3)
               kmeans_ar.fit(df[area_feature])
               df['cust_area_type'] = kmeans_ar.predict(df[area_feature])
               #-------------------------------------------------------------------------------
               kmeans_ad = KMeans(n_clusters=3)
               kmeans_ad.fit(df[address_feature])
               df['cust_address_type'] = kmeans_ad.predict(df[address_feature])
               #-------------------------------------------------------------------------------
               kmeans_c = KMeans(n_clusters=3)
               kmeans_c.fit(df[condition_feature])
               df['cust_condition_type'] = kmeans_c.predict(df[condition_feature])
               #-------------------------------------------------------------------------------
               df['sale_long']=df.apply(lambda r: sale_long(r['sale_yr'], r['sale_month'], r['sale_day']), axis=1)
               df['more_than_1_floor']=df.floors.apply(lambda x:1 if x>1 else 0)
               df['grade_more_than_8']=df.grade.apply(lambda x:1 if x>8 else 0)
               df['basement_or_not']=df.sqft_basement.apply(lambda x:1 if x>0 else 0)
               df['renovate_or_not']=df.yr_renovated.apply(lambda x:1 if x>0 else 0)
               df['built_long']=df.apply(lambda r: built_long(r['yr_built']), axis=1)


      2. The results :
         
         >> 。 Trained model based on the origin features : the best average validation loss = 29.51764771807034 
         >>
         >> 。 Trained model based on the origin and new features : the best average validation loss = 35.014327134428505

   。 Feature selection :

      I also do the feature selection through the correlation between features and price. The followings show both the confusion matrix based on the value of correlation and the results based on 'features that correlation more than 0.3', 'features that correlation more than 0.1', and 'all features'. However, the results show that it do not improve the performance; instead, it leads to worse performance.

   1. The confusion matrix based on the value of correlation
   
![Figure 2023-09-29 170318](https://github.com/Machine-Learning-NYCU/regression-house-sale-price-prediction-challenge-310581003/assets/145360863/780948b5-92ab-43fd-86d0-cee218db4ad2)

   2. The results :

      >> 。 Trained model based on all the origin features : the best average validation loss = 29.51764771807034
      >>
      >> 。 Trained model based on the origin features that the correlation more than 0.1 : the best average validation loss = 36.69512450832948
      >>
      >> 。 Trained model based on the origin features that the correlation more than 0.3 : the best average validation loss = 41.04318024062934
      >>
      >> 。 Trained model based on all the origin and new features : the best average validation loss = 35.014327134428505
      >>
      >> 。 Trained model based on the origin and new features that the correlation more than 0.1 : the best average validation loss = 37.05450312355391
      >>
      >> 。 Trained model based on the origin and new features that the correlation more than 0.3 : the best average validation loss = 41.18154934058306
