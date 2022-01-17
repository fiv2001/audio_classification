import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter


def train(model, epoch, train_loader, device, optimizer, criterion, log_interval):
    model.train()

    loss_sum = 0
    loss_cnt = 0
    for batch_idx, (sample_rate, inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss
        loss_cnt += 1

        iteration = epoch * len(train_loader) + batch_idx
        if batch_idx % log_interval == 0: #print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss))
            print('loss, iteration: ', loss, iteration)
            print('learning rate: , ', optimizer.param_groups[0]['lr'], iteration)
    
    avg_loss = loss_sum / loss_cnt
    writer = SummaryWriter('./runs/')
    writer.add_scalar('Loss/Train', avg_loss, epoch)


def test(model, epoch, test_loader, device, config):
    model.eval()
    class_correct = list(0. for i in range(config.getint('classes_num')))
    class_total = list(0. for i in range(config.getint('classes_num')))
    with torch.no_grad():
        for idx, (sample_rate, inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels)
            for i in range(len(inputs)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

    total_accuracy = 100 * sum(class_correct)/sum(class_total)
    print('[Iteration {}] Accuracy on the {} test images: {}%\n'.format(epoch, sum(class_total), total_accuracy))
    print('accuracy: ', total_accuracy)
    writer = SummaryWriter('./runs/')
    writer.add_scalar('Accuracy/Test', total_accuracy, epoch)
