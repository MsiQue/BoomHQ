import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def eval_model(F, model, loader, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = F(model, inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return 100 * correct / total, total_loss / len(loader)

def supervised_learning(F, model, train_dataset, val_dataset, test_dataset,
                                 criterion=nn.CrossEntropyLoss(),
                                 optimizer_class=optim.Adam,
                                 learning_rate=0.001,
                                 num_epochs=1000,
                                 batch_size=32):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = F(model, inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_accuracy, _ = eval_model(F, model, train_loader, criterion)
        val_accuracy, val_loss = eval_model(F, model, val_loader, criterion)
        print(f'Epoch {epoch + 1}/{num_epochs} : Train Accuracy: {train_accuracy:20}%  |  Validation Accuracy: {val_accuracy:20}%  |  Train Loss {running_loss / len(train_loader):20}  |  Val Loss {val_loss:20}')

        if val_accuracy > best_val_accuracy:
            best_epoch = epoch + 1
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)
    test_accuracy, test_loss = eval_model(F, model, test_loader, criterion)
    print(f'Accuracy on test set: {test_accuracy:20}%, Test Loss {test_loss:20}, Best Epoch {best_epoch:20}')

    return model

def supervised_learning_regression(F, model, train_dataset, val_dataset, test_dataset,
                                 criterion=nn.MSELoss(),
                                 optimizer_class=optim.Adam,
                                 learning_rate=0.001,
                                 num_epochs=2000,
                                 batch_size=32):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    best_val_loss = 1e99
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = F(model, inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader)}')

        val_loss = eval_model_regression(F, model, val_loader)
        print(f'Validation Loss: {val_loss}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        if val_loss < 1:
            break

    model.load_state_dict(best_model_state)
    test_loss = eval_model_regression(F, model, test_loader)
    print(f'Test Loss: {test_loss}%')

    return model