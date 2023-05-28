import os
import time
import torch
import copy
from util import plot_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = torch.nn.CrossEntropyLoss()


def train(args, model, optimizer, dataloaders):
    trainloader, testloader = dataloaders
    labs=[]
    best_testing_accuracy = 0.0

    # training
    for epoch in range(args.epochs):
        model.train()

        batch_time = time.time();
        iter_time = time.time()
        for i, data in enumerate(trainloader):

            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)

            cls_scores = model(imgs, with_dyn=args.with_dyn)
            loss = criterion(cls_scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.with_dyn == 1 and epoch == args.epochs-1 and i < 64:
                lab = labels[i].to(device)
                if 0 <= lab < 10 and (lab not in labs):
                    labs.append(lab)
                    print(labs)
                    kernels = model.dc.weight.data.detach().clone()
                    plot_weights(kernels)

            if i % 100 == 0 and i != 0:
                print('epoch:{}, iter:{}, time:{:.2f}, loss:{:.5f}'.format(epoch, i,
                                                                           time.time() - iter_time, loss.item()))
                iter_time = time.time()
        batch_time = time.time() - batch_time
        print('[epoch {} | time:{:.2f} | loss:{:.5f}]'.format(epoch, batch_time, loss.item()))
        print('-------------------------------------------------')

        if epoch % 1 == 0:
            testing_accuracy = evaluate(args, model, testloader)
            print('testing accuracy: {:.3f}'.format(testing_accuracy))

            if testing_accuracy > best_testing_accuracy:
                ### compare the previous best testing accuracy and the new testing accuracy
                ### save the model and the optimizer --------------------------------
                best_testing_accuracy = testing_accuracy  # Change the best test id
                checkpoint_path = './{}_checkpoint.pth'.format(args.exp_id)
                # Save learnable parameters and optimizer
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                # Save entire model - will not use as not recommended
                # torch.save(model, checkpoint_path)
                print('new best model saved at epoch: {}'.format(epoch))
    print('-------------------------------------------------')
    print('best testing accuracy achieved: {:.3f}'.format(best_testing_accuracy))


def evaluate(args, model, testloader):
    model.eval()
    total_count = torch.tensor([0.0]);
    correct_count = torch.tensor([0.0])
    for i, data in enumerate(testloader):
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)

        total_count += labels.size(0)

        with torch.no_grad():
            cls_scores = model(imgs, with_dyn=args.with_dyn)

            predict = torch.argmax(cls_scores, dim=1)
            correct_count += (predict == labels).sum()
    testing_accuracy = correct_count / total_count
    model.train()
    return testing_accuracy.item()


def resume(args, model, optimizer):
    checkpoint_path = './{}_checkpoint.pth'.format(args.exp_id)
    assert os.path.exists(checkpoint_path), ('checkpoint do not exits for %s' % checkpoint_path)

    ### load the model and the optimizer --------------------------------
    checkpoint = torch.load(checkpoint_path)
    # Loading learnable parameters and optimizer
    model.load_state_dict(copy.deepcopy(checkpoint['model_state_dict']))
    optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer_state_dict']))
    # Loading entire model - will not use as not recommended
    # model = torch.load(checkpoint_path)
    model.to(device)

    print('Resume completed for the model\n')

    return model, optimizer
