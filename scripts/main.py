import argparse
from functools import partial
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import *
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import training
import preprocess
import training_dataset
import rnn_models
import backtest
from pathlib import Path
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import mlflow.tensorflow
# import dvc.api
# from ray import tune
# from ray import train
# from ray.train import Checkpoint, get_checkpoint
# from ray.tune.schedulers import ASHAScheduler
# import ray.cloudpickle as pickle

# Initialize MLflow tracking
mlflow.set_experiment("Test experiment")

parser = argparse.ArgumentParser(description='LOB RNN Model: Main Function')
parser.add_argument('--data_file', type=str, default='../data',
                    help='location of market data')
parser.add_argument('--backtesting_files', nargs='+', type=str, default=['backtest.csv'],
                    help='location of backtesting data (can pass multiple files)')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of rnn (LSTM, GRU)')
parser.add_argument('--hidden_size', type=int, default=128,
                    help='hidden_size in rnn model')
parser.add_argument('--num_layers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--fc_layers_size', type=int, default=100,
                    help='size of fully connected layers neurons')
parser.add_argument('--predict_events', type=int, default=1,
                    help='the event in the future to predict')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--num_epochs', type=int, default=50,
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=512,
                    help='batch size')
parser.add_argument('--sequence_length', type=int, default=60,
                    help='number of events in the sequence')
parser.add_argument('--val_interval', type=int, default=10,
                    help='validation interval')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--save', type=str,  default='model.pth',
                    help='path to save the final model')
args = parser.parse_args()

def main():
    with mlflow.start_run() as run:
        mlflow.log_params(vars(args))  # Log parameters
        
        torch.manual_seed(args.seed)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        df = pd.read_csv(args.data_file)
        
        print('Preprocessing data....')
        new_df = preprocess.process_data(df)
        dataset = training_dataset.OrderBookDataset(new_df, args.sequence_length, args.predict_events)
        print('Data preprocessing complete')
        train_loader, val_loader, test_loader = training_dataset.get_data_loaders(dataset, 0.8, args.batch_size)
        if args.model == 'LSTM':
            model = rnn_models.LSTMModel(len(dataset[0][0][0]),args.hidden_size ,args.num_layers, args.fc_layers_size).to(device)
        elif args.model == 'GRU':
            model = rnn_models.GRUModel(len(dataset[0][0][0]),args.hidden_size ,args.num_layers, args.fc_layers_size).to(device)
        else:
            raise ValueError('Model not supported')

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        def train():
            # Train the model in one epoch
            model.train()
            train_loss = 0
            for i, (x, y) in enumerate(tqdm(train_loader, desc="Training")):
                x = x.float()  # Convert input data to torch.float32 type
                y = y.long()  # Convert target data to torch.float32 type
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_hat = model(x)
                y_hat = y_hat.to(torch.float32)
                loss = criterion(y_hat, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                train_loss += loss.item()
            train_loss = train_loss / len(train_loader)
            return train_loss

        def evaluate():
            model.eval()
            model.eval()
            all_y_true = torch.LongTensor()
            all_y_pred = torch.LongTensor()
            val_loss = 0
            for x, y in val_loader:
                x = x.float()  # Convert input data to torch.float32 type
                y = y.long()  # Convert target data to torch.float32 type
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                val_loss += loss.item()
                y_hat = F.softmax(y_hat, dim=1)
                y_pred = torch.argmax(y_hat, dim=1)
                all_y_true = torch.cat((all_y_true, y.to('cpu').long()), dim=0)
                all_y_pred = torch.cat((all_y_pred,  y_pred.to('cpu').long()), dim=0)
            val_loss = val_loss / len(val_loader)
            acc,  precision, recall, f1 = classification_metrics(all_y_pred.detach().numpy(), 
                                                                    all_y_true.detach().numpy())
            return val_loss, acc, precision, recall, f1

        def classification_metrics(Y_pred, Y_true):
            acc, precision, recall, f1score = accuracy_score(Y_true, Y_pred), \
                                                precision_score(Y_true, Y_pred, average='weighted', zero_division = 1), \
                                                recall_score(Y_true, Y_pred, average='weighted'), \
                                                f1_score(Y_true, Y_pred, average='weighted')
            return acc,  precision, recall, f1score

        try:
            print('---- Start of training ----')
            train_loss_log = []
            val_loss_log = []
            min_val_loss = None
            total_training_time = 0
            start_epoch = 1
            for epoch in range(start_epoch, args.num_epochs+1):
                start_time = time.time()
                train_loss = train()
                end_time = time.time()      
                epoch_duration = end_time - start_time
                total_training_time += epoch_duration

                train_loss_log.append(train_loss)

                if epoch % args.val_interval == 0:
                    val_loss, acc, precision, recall, f1 = evaluate()
                    val_loss_log.append(val_loss)
                    if not min_val_loss or val_loss < min_val_loss:
                        mlflow.pytorch.log_model(model, "model")  # Log the best model
                        torch.save(model.state_dict(), args.save)
                        min_val_loss = val_loss
                    # Log metrics
                    mlflow.log_metrics({"val_loss": val_loss, "accuracy": acc, "precision": precision, "recall": recall, "f1_score": f1})

                    print(f"Epoch: {epoch} \tTime: {epoch_duration:.2f}s \tTraining Loss: {train_loss:.6f} Validation Loss: {val_loss:.6f} acc: {acc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}")
                else:
                    print(f"Epoch: {epoch} \tTime: {epoch_duration:.2f}s \tTraining Loss: {train_loss:.6f} ")

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
            print('=' * 89)

        print('---- End of training ----')
        _, acc, precision, recall, f1score = training.evaluate(model, test_loader)
        hours, remainder = divmod(total_training_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print('=' * 89)
        print('---- End of training ----')
        print(f" Total training time: {int(hours)}h:{int(minutes)}m:{seconds:.2f}s \tacc: {acc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1score:.3f}")
        print('=' * 89)
        plt.figure()
        plt.plot(train_loss_log, label='Training loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
        plt.plot(val_loss_log, label='Validation loss')
        plt.title('Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        print("----- Beginning backtesting -----")
        print('Starting Portfolio Value: 100000')
        print('=' * 89)
            
        final_value = backtest.backtest_model(model, args.backtesting_files, device, args.sequence_length)
        # Log model artifacts
        mlflow.log_artifact(args.save, "model.pth")

if  __name__ == '__main__':
    main()

