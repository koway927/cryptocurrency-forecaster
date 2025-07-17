import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, val_loader, num_epochs, learning_rate, save, weight = None):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight = weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            #print(x.shape)
            x = x.float()  # Convert input data to torch.float32 type
            y = y.long()  # Convert target data to torch.float32 type

            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            #print(x.shape, y.shape, y_hat.shape)
            y_hat = model(x)
            y_hat = y_hat.to(torch.float32)
 
            #y_hat = y_hat.float(requires_grad=True)
            #print(y_hat.shape)
            #print(y_pred)
            loss = criterion(y_hat, y)
            #print(loss)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
                
        train_loss = train_loss / len(train_loader)
        torch.save(model.state_dict(), save)
        if epoch % 10 == 0:
            val_loss, acc, precision, recall, f1 = evaluate(model, val_loader, weight = None)
            print(f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} Validation Loss: {val_loss:.6f} acc: {acc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}")

def evaluate(model, loader, weight = None):
    model.eval()
    criterion = nn.CrossEntropyLoss(weight = weight)
    all_y_true = torch.LongTensor()
    all_y_pred = torch.LongTensor()
    all_y_score = torch.FloatTensor()
    val_loss = 0
    for x, y in loader:
        x = x.float()  # Convert input data to torch.float32 type
        y = y.long()  # Convert target data to torch.float32 type
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        val_loss += loss.item()
        y_hat = F.softmax(y_hat, dim=1)
        y_pred = torch.argmax(y_hat, dim=1)
        """print("y",y)
        print("y_hat",y_hat)
        print("y_pred",y_pred)"""
        all_y_true = torch.cat((all_y_true, y.to('cpu').long()), dim=0)
        all_y_pred = torch.cat((all_y_pred,  y_pred.to('cpu').long()), dim=0)
        all_y_score = torch.cat((all_y_score,  y_hat.to('cpu')), dim=0)
    val_loss = val_loss / len(loader)
    acc,  precision, recall, f1 = classification_metrics(all_y_score.detach().numpy(), 
                                                             all_y_pred.detach().numpy(), 
                                                             all_y_true.detach().numpy(), weight)
    #print(f"acc: {acc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}")
    return val_loss, acc, precision, recall, f1


def classification_metrics(Y_score, Y_pred, Y_true, weight = None):
    acc, precision, recall, f1score = accuracy_score(Y_true, Y_pred, sample_weight= weight), \
                                           precision_score(Y_true, Y_pred, average='weighted', zero_division = 1), \
                                           recall_score(Y_true, Y_pred, average='weighted'), \
                                           f1_score(Y_true, Y_pred, average='weighted', sample_weight= weight)
    return acc,  precision, recall, f1score
