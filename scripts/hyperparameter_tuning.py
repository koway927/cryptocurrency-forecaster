import argparse
import os
import tempfile
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
from pathlib import Path
import ray
from ray import tune
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import main
import hashlib
parser = argparse.ArgumentParser(description='LOB RNN Model: Main Function')
parser.add_argument('--data_file', type=str, default='../data',
                    help='location of market data')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of rnn (LSTM, GRU)')
parser.add_argument('--num_epochs', type=int, default=50,
                    help='number of epochs')
parser.add_argument('--sequence_length', type=int, default=60,
                    help='number of events in the sequence')
parser.add_argument('--predict_events', type=int, default=1,
                    help='the event in the future to predict')
parser.add_argument('--val_interval', type=int, default=10,
                    help='validation interval')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--save', type=str,  default='model.pth',
                    help='path to save the final model')
args = parser.parse_args()

def trial_dirname_creator(trial):
    # Use the trial's unique ID and a hash of its parameters to create a shorter directory name
    param_hash = hashlib.md5(str(trial.config).encode()).hexdigest()[:6]
    return f"{trial.trainable_name}_{trial.trial_id}_{param_hash}"

def classification_metrics(Y_pred, Y_true):
    acc, precision, recall, f1score = accuracy_score(Y_true, Y_pred), \
                                        precision_score(Y_true, Y_pred, average='weighted', zero_division = 1), \
                                        recall_score(Y_true, Y_pred, average='weighted'), \
                                        f1_score(Y_true, Y_pred, average='weighted')
    return acc,  precision, recall, f1score
        
def evaluate(model ,val_loader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def train_rnn(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #Laod and precoess data
    df = pd.read_csv(args.data_file)
    print('Preprocessing data....')
    new_df = preprocess.process_data(df)
    dataset = training_dataset.OrderBookDataset(new_df, args.sequence_length, args.predict_events)
    print('Data preprocessing complete')
    train_loader, val_loader, _ = training_dataset.get_data_loaders(dataset, 0.8, config["batch_size"])
    if args.model == 'LSTM':
        model = rnn_models.LSTMModel(len(dataset[0][0][0]),config["hidden_size"] ,config["num_layers"], config["fully_connected_layers_neurons"]).to(device)
    elif args.model == 'GRU':
        model = rnn_models.GRUModel(len(dataset[0][0][0]),config["hidden_size"] ,config["num_layers"], config["fully_connected_layers_neurons"]).to(device)
    else:
        raise ValueError('Model not supported')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    checkpoint = get_checkpoint()
    print("checkpoint")
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 1

    
    train_loss_log = []
    val_loss_log = []
    min_val_loss = None
    total_training_time = 0
    for epoch in range(start_epoch, args.num_epochs+1):

        start_time = time.time()
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
        end_time = time.time()      
        epoch_duration = end_time - start_time
        total_training_time += epoch_duration
        
        train_loss_log.append(train_loss)
        val_loss, acc, precision, recall, f1 = evaluate(model, val_loader, criterion)
        val_loss_log.append(val_loss)
        if not min_val_loss or val_loss < min_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            min_val_loss = val_loss
                
        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            ray.train.report(
                {"loss": val_loss, "accuracy": acc},
                checkpoint=checkpoint,
            )
        print(f"Epoch: {epoch} \tTime: {epoch_duration:.2f}s \tTraining Loss: {train_loss:.6f} Validation Loss: {val_loss:.6f} acc: {acc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}")

def main():
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    config = {
        "fully_connected_layers_neurons": tune.choice([2 ** i for i in range(3)]),
        "hidden_size": tune.choice([2 ** i for i in range(3)]),
        "num_layers": tune.choice([ i for i in range(1, 11)]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2 ** i for i in range(6, 10)]),
    }
    
    scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=args.num_epochs,
            grace_period=1,
            reduction_factor=2,
        )
    
    print('---- Start of training ----')
    results = tune.run(
        partial(train_rnn),
        resources_per_trial={"cpu": 16, "gpu": 1},
        config=config,
        num_samples=10,
        scheduler=scheduler,
        trial_dirname_creator=trial_dirname_creator
    )
    
    best_result = results.get_best_trial("loss", "min")
    
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    print('---- End of training ----')
    
    df = pd.read_csv(args.data_file)
    new_df = preprocess.process_data(df)
    dataset = training_dataset.OrderBookDataset(new_df, args.sequence_length, args.predict_events)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model == 'LSTM':
        best_trained_model = rnn_models.LSTMModel(len(dataset[0][0][0]),best_result.config["hidden_size"] ,best_result.config["num_layers"], best_trial.config["fully_connected_layers_neurons"]).to(device)
    elif args.model == 'GRU':
        best_trained_model = rnn_models.GRUModel(len(dataset[0][0][0]),best_result.config["hidden_size"] ,best_result.config["num_layers"], best_trial.config["fully_connected_layers_neurons"]).to(device)
    else:
        raise ValueError('Model not supported')
    
    _, _, test_loader = training_dataset.get_data_loaders(dataset, 0.8, best_result.config["batch_size"])
    
    best_trained_model.to(device)

    best_checkpoint = results.get_best_checkpoint(trial=best_result, metric="accuracy", mode="max")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        _ , acc, precision, recall, f1score = training.evaluate(best_trained_model, test_loader)
        print("Best trial test set accuracy: {}".format(acc))
        print('---- End of training ----')

if  __name__ == '__main__':
    main()
