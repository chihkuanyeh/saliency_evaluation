import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import loader


def evaluate(data_loader, model):
    model.eval()
    test_loss = 0.
    correct = 0.
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            X, y = X.cuda(), y.cuda()
            out = model(X)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(out, y).item()
            pred = out.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)
    print('Test Acc: ', acc, 'Test Loss: ', test_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--load')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = loader.mnist_loaders(args.batch_size)

    if os.path.splitext(args.load)[1] == '.npz':
        model, _ = loader.mnist_load_model(args.load, state_dict=True, tf=True)
    else:
        model, _ = loader.mnist_load_model(args.load, state_dict=True, tf=False)
    model = model.cuda()
    evaluate(test_loader, model)
