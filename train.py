import torch
import sys
import os
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import *
import json
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    batch_size = 64
    train_dataset = torchvision.datasets.CIFAR10("../data", train=True, transform=data_transform["train"], download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=data_transform["val"], download=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
    train_num = len(train_dataset)
    val_num = len(validate_dataset)
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    net = resnet101(num_classes=10)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 30
    best_acc = 0.0
    save_path = './resNet101.pth'
    train_steps = len(train_loader)

    writer = SummaryWriter("../logs2")

    train_steps = 1
    val_acc_steps = 1
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            writer.add_scalar("train_loss", loss, train_steps)
            train_steps += 1

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        writer.add_scalar("val_acc", val_accurate, val_acc_steps)
        val_acc_steps += 1
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')
    writer.close()

    with open('accuracy.txt', 'w') as file:
        # 这是你的准确率值，你可以替换成你的实际准确率
        file.write('Accuracy: {:.2f}%'.format(best_acc * 100))


if __name__ == '__main__':
    main()
