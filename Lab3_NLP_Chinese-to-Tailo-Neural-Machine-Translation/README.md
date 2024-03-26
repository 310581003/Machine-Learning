# Lab3_NLP_Chinese-to-Tailo-Neural-Machine-Translation

Goal
-
The main purpose of this experiment is to establish a Chinese-to-Tailo translation machine and an example is showed as followings. Moreover, the competition is on Kaggle (https://www.kaggle.com/competitions/machine-learning-2023nycu-translation/submissions).

(ex.)
- Input sentence : Obama大勝美國頭一位烏人總統。
- Target sentence : Obama tua7-sing3 bi2-kok4 thau5-tsit8-ui7 oo1-lang5 tsong2-thong2。
  

File Discription
-
1. code
  - trainer.py : Main code for training and testing.
  - trainer.ipynb : Main code on Colab for training and testing.
2. Result
  - record.txt : The record of training processing.
  - test.csv : Predictions based on testing dataset.
  - Learning curve.png : The visualization figure of learning curve based on training and validation loss.

  ------------------------------------------------------------
  Due to the file size limitations on GitHub, you can access the following files via the provided cloud link: 
  https://drive.google.com/drive/folders/1dwgqd7fX_-tPEZb0g2V6yN7822_YVd97?usp=drive_link
  
  - Best_Model.pth : Saved model based on the best validation accuracy.
  -  Model.pth : The lastest saved model.
  -  Learning curve.png : The visualization figure of learning curve based on training and validation loss.
  

How to start up
-

Please open file trainer.py. And in the following block, def parse_config():, you can set the path of the folders for training data and testing data respectively. Moreover, you can modify the values of hyperparameters and the path of output files by your own. After setting these parameters, you can run this code to start training.

    ###########################################################################
    ##                      Argparse Setting                                 ##
    ###########################################################################
    def parse_config():
      parser = argparse.ArgumentParser()
      parser.add_argument("--train_data", default='/content/train-ZH.csv', help="path of training data")
      parser.add_argument("--train_target", default='/content/train-TL.csv', help="path of training target")
      parser.add_argument("--test_data", default='/content/drive/MyDrive/Colab Notebooks/2023ML/Lab3/Data/test-ZH-nospace.csv', help="path of testing data")

      parser.add_argument("--n_epochs", type=int, default=120, help="number of epochs of training")
      parser.add_argument("--batch_size", type=int, default=1000, help="size of the batches")
      parser.add_argument("--hidden_size", type=int, default=512, help="hidden size of RNN")
      parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
      parser.add_argument("--opt",  default='Adam', help="Optimizer:'SGD','RMSprop','Adagrad','Adam'")
      parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
      parser.add_argument("--b2", type=float, default=0.97, help="adam: decay of second order momentum of gradient")
      parser.add_argument("--loss", default='Cross_entropy', help="Loss func:'hinge_loss','MAE','MSE','NLL(Negative Log-Likelihood)','Cross_entropy'")
      parser.add_argument("--l2_reg", type=float, default=0.0006, help="l2 regularization")
      parser.add_argument("--val_split", type=float, default=0.2, help="validation split")

      parser.add_argument('--outf', default='/content/drive/MyDrive/Colab Notebooks/2023ML/Lab3/Result/', help='folder to output record and model')
      parser.add_argument('--log', default='/content/drive/MyDrive/Colab Notebooks/2023ML/Lab3/Result/record.txt', help='path to record')
      parser.add_argument('--font', default=20, help='font size for output figure')

      args = parser.parse_args(args=[])
      return args

Methodology
-
![Process](https://github.com/Machine-Learning-NYCU/chinese-to-tailo-neural-machine-translation-310581003/assets/145360863/0dac4b0e-3c89-4bec-9196-91ddcf143edd)

1. Word2Vector

     In order to use text as input for the model's learning process, in this experiment, we first break down each sentence in Chinese and Tailo into individual characters. Then, these characters are encoded to establish a directory mapping each character to its encoding. Below is the code implementing Word2Vector.

    | Training data :
   
     - Chinese : 63469 sentences.
     - Tailo : 63469 sentences.  
  
    | Testing data : 

     - Chinese : 641 sentences.
   
    | Code implementing Word2Vector (ref: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

        SOS_token = 0
        EOS_token = 1
        MAX_LENGTH = 60
        args = parse_config()
        class Lang:
            def __init__(self, name):
                self.name = name
                self.word2index = {}
                self.word2count = {}
                self.index2word = {0: "SOS", 1: "EOS"}
                self.n_words = 2  # Count SOS and EOS

            def addSentence(self, sentence):
                if self.name == 'talo':
                  for word in nlp(sentence):
                      self.addWord(str(word))
                else:
                  for word in sentence.replace(" ", ""):
                      self.addWord(word)

            def addWord(self, word):
                  if word not in self.word2index:
                      self.word2index[word] = self.n_words
                      self.word2count[word] = 1
                      self.index2word[self.n_words] = word
                      self.n_words += 1
                  else:
                      self.word2count[word] += 1


        def readLangs(lang1, lang2, reverse=False):
             print("Reading lines...")

            # Read the file and split into lines
            lines_lang1 = pd.read_csv(args.train_data, usecols=["txt"], encoding='utf-8').values.tolist()
            lines_lang2 = pd.read_csv(args.train_target, usecols=["txt"], encoding='utf-8').values.tolist()

            # Reverse pairs, make Lang instances
            input_lang = Lang(lang1)
            output_lang = Lang(lang2)
            return input_lang, output_lang, lines_lang1, lines_lang2

        def filterPair(p):
            return len(p.split(' ')) < MAX_LENGTH


        def filterPairs(pairs):
            return [pair for pair in pairs if filterPair(pair)]

        def prepareData(lang1, lang2, reverse=False):
            input_lang, output_lang, lines_lang1, lines_lang2 = readLangs(lang1, lang2, reverse)
            print("Read %s sentences of input_lang" % len(lines_lang1))
            print("Read %s sentences of output_lang" % len(lines_lang2))
            with open(args.log, 'a') as f:
                f.write(f'------------------------------------------\n')
                f.write(f'-     Data pre-processing     -\n')
                f.write(f'------------------------------------------\n')
                f.write(">>> Reading data ...\n")
                f.write("Read %s sentences of input_lang (Chinese)\n" % len(lines_lang1))
                f.write("Read %s sentences of output_lang (Tailo)\n" % len(lines_lang2))

            print("Counting words...")
            for pair in lines_lang1:
                input_lang.addSentence(pair[0])

            lines_lang1_test = pd.read_csv(args.test_data, usecols=["txt"], encoding='utf-8').values.tolist()
            for pair in lines_lang1_test:
                input_lang.addSentence(pair[0])

            for pair in lines_lang2:
                output_lang.addSentence(pair[0])
            print("Counted words:")
            print(input_lang.name, input_lang.n_words)
            print(output_lang.name, output_lang.n_words)
            with open(args.log, 'a') as f:
                f.write("- Counted words :\n")
                f.write(f'Chinese : {input_lang.n_words} words in total\n')
                f.write(f'Tailo : {output_lang.n_words} words in total\n')
            return input_lang, output_lang, lines_lang1, lines_lang2


        def indexesFromSentence(lang, sentence):
            if lang.name == 'talo':
                return [lang.word2index[str(word)] for word in nlp(sentence)]
            else :
                return [lang.word2index[word] for word in sentence.replace(" ", "")]

        def tensorFromSentence(lang, sentence):

            indexes = indexesFromSentence(lang, sentence)
            indexes.append(EOS_token)
            return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

        def tensorsFromPair(pair):
            input_tensor = tensorFromSentence(input_lang, pair[0])
            target_tensor = tensorFromSentence(output_lang, pair[1])
            return (input_tensor, target_tensor)
                 

2. Hyperparameters :
    - Epoch : 100
    - Optimizer : Adam
    - Learning rate : 0.0001
    - Loss function : Cross entropy
    - Batch size : 700

Code Discriptions for trainer.py
-

1. Argparse Settings

    Initial variable setting including the path to input & output files, hyperparameter settings.

        ###########################################################################
        ##                      Argparse Setting                                 ##
        ###########################################################################
        def parse_config():
            parser = argparse.ArgumentParser()
            parser.add_argument("--train_data", default='/content/train-ZH.csv', help="path of training data")
            parser.add_argument("--train_target", default='/content/train-TL.csv', help="path of training target")
            parser.add_argument("--test_data", default='/content/drive/MyDrive/Colab Notebooks/2023ML/Lab3/Data/test-ZH-nospace.csv', help="path of testing data")

            parser.add_argument("--n_epochs", type=int, default=120, help="number of epochs of training")
            parser.add_argument("--batch_size", type=int, default=1000, help="size of the batches")
            parser.add_argument("--hidden_size", type=int, default=512, help="hidden size of RNN")
            parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
            parser.add_argument("--opt",  default='Adam', help="Optimizer:'SGD','RMSprop','Adagrad','Adam'")
            parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
            parser.add_argument("--b2", type=float, default=0.97, help="adam: decay of second order momentum of gradient")
            parser.add_argument("--loss", default='Cross_entropy', help="Loss func:'hinge_loss','MAE','MSE','NLL(Negative Log-Likelihood)','Cross_entropy'")
            parser.add_argument("--l2_reg", type=float, default=0.0006, help="l2 regularization")
            parser.add_argument("--val_split", type=float, default=0.2, help="validation split")

            parser.add_argument('--outf', default='/content/drive/MyDrive/Colab Notebooks/2023ML/Lab3/Result/', help='folder to output record and model')
            parser.add_argument('--log', default='/content/drive/MyDrive/Colab Notebooks/2023ML/Lab3/Result/record.txt', help='path to record')
            parser.add_argument('--font', default=20, help='font size for output figure')

            args = parser.parse_args(args=[])
            return args

2. Build the training dataset

   To perform Word2Vector on the training data and to split training data into training dataset and validation dataset.

         def get_dataloader():
            args = parse_config()
            input_lang, output_lang, lines_lang1, lines_lang2 = prepareData('ch', 'talo', True)

            n = len(lines_lang1)
            print(f'length of lang1 : {len(lines_lang1)}')
            print(f'length of lang2 : {len(lines_lang2)}')
            assert len(lines_lang1)==len(lines_lang2)
            input_ids = np.ones((n, MAX_LENGTH), dtype=np.int32)
            target_ids = np.ones((n, MAX_LENGTH), dtype=np.int32)

            idx = 0
            for sent in lines_lang1:
                inp_ids = indexesFromSentence(input_lang, sent[0])
                inp_ids.append(EOS_token)

                input_ids[idx, :len(inp_ids)] = inp_ids
                idx = idx+1

            idx = 0
            for sent in lines_lang2:
                tgt_ids = indexesFromSentence(output_lang, sent[0])
                tgt_ids.append(EOS_token)

                target_ids[idx, :len(tgt_ids)] = tgt_ids
                idx = idx+1

            train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                                       torch.LongTensor(target_ids).to(device))

            train_size = int(0.8 * len(train_data))
            valid_size = len(train_data) - train_size
            train_dataset, valid_dataset = random_split(train_data, [train_size, valid_size])
            train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
            valid_loader= DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True)
            print(f'word2index of ch : {input_lang.word2index}')
            print(f'index2word of tailo : {output_lang.index2word}')
            with open(args.log, 'a', encoding='UTF-8') as f:
                 f.write(f'>Building directory...\n')
                 f.write(f'word2index of chinese : {input_lang.word2index}\n')
                 f.write(f'index2word of tailo : {output_lang.index2word}\n')
            return input_lang, output_lang, train_loader, valid_loader

3. Definition of model 
   
    Definitions of encoder and decoder. (ref: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
   
         ###########################################################################
        ##                            Model                                      ##
        ###########################################################################
        class EncoderRNN(nn.Module):
            def __init__(self, input_size, hidden_size, dropout_p=0.1):
                super(EncoderRNN, self).__init__()
                self.hidden_size = hidden_size

                self.embedding = nn.Embedding(input_size, hidden_size)
                self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
                self.dropout = nn.Dropout(dropout_p)

            def forward(self, input):
                embedded = self.dropout(self.embedding(input))
                output, hidden = self.gru(embedded)
                return output, hidden

        class BahdanauAttention(nn.Module):
            def __init__(self, hidden_size):
                super(BahdanauAttention, self).__init__()
                self.Wa = nn.Linear(hidden_size, hidden_size)
                self.Ua = nn.Linear(hidden_size, hidden_size)
                self.Va = nn.Linear(hidden_size, 1)

            def forward(self, query, keys):
                scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
                scores = scores.squeeze(2).unsqueeze(1)
                weights = F.softmax(scores, dim=-1)
                context = torch.bmm(weights, keys)

                return context, weights

        class AttnDecoderRNN(nn.Module):
            def __init__(self, hidden_size, output_size, dropout_p=0.1):
                super(AttnDecoderRNN, self).__init__()
                self.embedding = nn.Embedding(output_size, hidden_size)
                self.attention = BahdanauAttention(hidden_size)
                self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
                self.out = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(dropout_p)

            def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
                batch_size = encoder_outputs.size(0)
                decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)

                decoder_hidden = encoder_hidden
                decoder_outputs = []
                attentions = []

                for i in range(MAX_LENGTH):
                    decoder_output, decoder_hidden, attn_weights = self.forward_step(
                    decoder_input, decoder_hidden, encoder_outputs
                    )
                    decoder_outputs.append(decoder_output)
                    attentions.append(attn_weights)

                    if target_tensor is not None:
                        # Teacher forcing: Feed the target as the next input
                        decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
                    else:
                        # Without teacher forcing: use its own predictions as the next input
                        _, topi = decoder_output.topk(1)
                        decoder_input = topi.squeeze(-1).detach()  # detach from history as input

                decoder_outputs = torch.cat(decoder_outputs, dim=1)
                decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
                attentions = torch.cat(attentions, dim=1)

                return decoder_outputs, decoder_hidden, attentions


            def forward_step(self, input, hidden, encoder_outputs):
                embedded =  self.dropout(self.embedding(input))
                query = hidden.permute(1, 0, 2)
                context, attn_weights = self.attention(query, encoder_outputs)
                input_gru = torch.cat((embedded, context), dim=2)

                output, hidden = self.gru(input_gru, hidden)
                output = self.out(output)

                return output, hidden, attn_weights

4. Training & Validation

    Definitions of training loop and validation.
   
        ###########################################################################
        ##                           Training                                     ##
        ###########################################################################

        def train(encoder, decoder, train_loader, valid_loader, epochs, device):
         """
            Args:
                encoder: encoder model
                decoder: decoder model
                train_loader: training dataloader
                valid_loader: validation dataloader
                epochs: number of training epoch
                device: gpu/cpu
            Returns:
                df_loss : dataframe with column 'epoch','loss_train','loss_valid'
                best_model_wts : saved model based on the best loss
         """
            args = parse_config()
            df_loss=pd.DataFrame()
            df_loss['epoch']=range(1,epochs+1)
            best_model_wts=None
            best_evaluated_loss = 100000

            encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
            decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
            loss_func = nn.CrossEntropyLoss()

            acc_train=list()
            acc_valid=list()
            loss_train=list()
            loss_valid=list()

            with open(args.log, 'a') as f:
                f.write(f'------------------------------------------\n')
                f.write(f'-         Training       -\n')
                f.write(f'------------------------------------------\n')
                f.write(f'>>> Start training... \n')
            for epoch in range(1,epochs+1):
            """
              train
            """
                with torch.set_grad_enabled(True):

                    t_loss=0
                    t_acc=0
                    correct=0
                    for i, (data, labels) in enumerate(tqdm(train_loader, desc='Epoch '+str(epoch))):

                        data, labels=data.to(device),labels.to(device)
                        encoder_optimizer.zero_grad()
                        decoder_optimizer.zero_grad()
                        encoder_outputs, encoder_hidden = encoder(data)
                        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, labels)

                        topi = torch.argmax(decoder_outputs,dim=-1)
                        decoded_ids = topi.squeeze()

                        loss = loss_func(
                            decoder_outputs.view(-1, decoder_outputs.size(-1)),
                            labels.view(-1)
                        )

                        """
                          update
                        """

                        loss.backward()  # bp
                        encoder_optimizer.step()
                        decoder_optimizer.step()
                        t_loss+=loss.item()
                    t_loss/=len(train_loader.dataset)
                    loss_train.append(t_loss)

                """
                evaluate
                """
                # call function evaluate
                v_loss=evaluate(encoder, decoder, valid_loader, loss_func, device)
                loss_valid.append(v_loss)
                print(f'--Epoch{epoch:>2d} | Train loss:{t_loss:.6f} | Valid loss:{v_loss:.6f}\n')
                with open(args.log, 'a') as f:
                    f.write(f'--Epoch{epoch:>2d} | Train loss:{t_loss:.6f}| Valid loss:{v_loss:.6f}\n')

                # update best_model_wts
                if v_loss < best_evaluated_loss:
                    best_evaluated_acc=v_loss
                    best_model_wts={'encoder': encoder,
                                    'decoder': decoder}
                    torch.save(best_model_wts, os.path.join(args.outf, f'Best_Model.pt'))

                # saved model
                checkpoint_dict = {'last_epoch': epoch,
                                   'encoder': encoder.state_dict(),
                                   'decoder': decoder.state_dict(),
                                   'best_loss' : best_evaluated_loss,
                                   'loss_train' : loss_train,
                                   'loss_valid' : loss_valid}
                torch.save( checkpoint_dict, os.path.join(args.outf, f'Model.pt'))


            df_loss['loss_train']=loss_train
            df_loss['loss_valid']=loss_valid

            return df_loss, best_model_wts

        def evaluate(encoder, decoder, valid_loader, loss_func, device):
          """
            Args:
                encoder: encoder model
                decoder: decoder model
                valid_loader: validation dataloader
                loss_func : loss function
                device: gpu/cpu
            Returns:
                total_loss : validation average loss
          """

            args = parse_config()

            print(f'valid:{len(valid_loader)}')
            with torch.set_grad_enabled(False):

                correct=0
                total_loss=0
                acc=0
                for data, labels in valid_loader:
                    data,labels=data.to(device),labels.to(device)
                    encoder_outputs, encoder_hidden = encoder(data)
                    decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, labels)

                    topi = torch.argmax(decoder_outputs,dim=-1)
                    decoded_ids = topi.squeeze()

                    loss = loss_func(
                        decoder_outputs.view(-1, decoder_outputs.size(-1)),
                        labels.view(-1)
                    )

                    total_loss+=loss.item()

                total_loss/=len(valid_loader.dataset)
            return total_loss

5. Plotting
   
     To plot the learning curve for model assessment and return figures for saving.

       ###########################################################################
        ##                               Plot                                   ##
        ###########################################################################
        def plot(dataframe1, mode):
        """
          Arguments:
              dataframe1: dataframe with 'epoch','loss_train','loss_valid' columns
              mode : loss/accuracy
          Returns:
              figure: figure of learning curve
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

6. Testing
   
   Inference the translation for the testing data based on trained model and output the test.csv file.

         ###########################################################################
        ##                            Test                                       ##
        ###########################################################################
        def test(input_lang, encoder, decoder,outf):
          """
            Args:
                input_lang : directory of input language
                encoder: trained encoder model
                decoder: trained decoder model
                outf: path to output file
          """
            df_id =list()
            df_txt = list()
            test_lines = pd.read_csv(args.test_data, usecols=["txt"], encoding='utf-8').values.tolist()
            print("Read %s sentences of testing data" % len(test_lines))
            with open(args.log, 'a') as f:
              f.write(f'------------------------------------------\n')
              f.write(f'-        Testing        -\n')
              f.write(f'------------------------------------------\n')
              f.write(f'>> Testing data : {len(test_lines)}\n')
            with torch.set_grad_enabled(False):
                encoder.eval()
                decoder.eval()
                correct=0
                i = 1
                for data in test_lines:
                    input_tensor = tensorFromSentence(input_lang, data[0])
                    encoder_outputs, encoder_hidden = encoder(input_tensor)
                    decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)
            
                    topi = torch.argmax(decoder_outputs,dim=-1)
                    decoded_ids = topi.squeeze()
                    decoded_words = []
                    for idx in decoded_ids:
                        if idx.item() == EOS_token:
                            decoded_words.append('<EOS>')
                            break
                        decoded_words.append(output_lang.index2word[idx.item()])
                        output_sentence = ' '.join(decoded_words)
                    df_id.append(i)
                    df_txt.append(output_sentence)
                    i=i+1
            for words in df_txt:
              if words == ' - ':
                words = '-'
            df = pd.DataFrame()
            df['id'] = df_id
            df['txt'] = df_txt
            df.to_csv(os.path.join(outf, 'test.csv'), index=None)

Results & Disscusion
-

1. Results

   - Best validation :
       - Loss : 0.000103
         
   - Best testing on Kaggle :
       - Private score : 5.40104
    
   - Learning curve :
       - training loss versus validation loss

            ![Learning_curve](https://github.com/Machine-Learning-NYCU/chinese-to-tailo-neural-machine-translation-310581003/assets/145360863/7b9d65d4-ae1e-4236-8733-558834cae919)


  2. Disscussion

     - Compare to the results of applying different settings

          This experiment also attempted to use a larger batch size and more epochs for training. Despite achieving a lower validation loss (0.000084), the predicted results yielded a lower Kaggle private score of 15.03271, performing worse. The detailed hyperparameter settings are as follows:

         - Epoch : 120
        - Optimizer : Adam
        - Learning rate : 0.0001
        - Loss function : Cross entropy
        - Batch size : 1000



        

