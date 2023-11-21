import torch


def train_one_epoch(tr_loader, model, loss_func, optimizer, device):
    running_loss = 0.0

    for i, data in enumerate(tr_loader):
        im1,im2,im3 = data['images']
        y = data['joint_values'].float()
        im1,im2,im3, y = im1.to(device), im2.to(device), im3.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(im1, im2, im3).float()
        loss = loss_func(preds, y)
        running_loss +=loss.item()
        loss.backward()
        optimizer.step()

    train_loss = running_loss/(i+1)

    return(train_loss)

def validate_one_epoch(val_loader, model, loss_func, optimizer, device):
    running_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            im1,im2,im3 = data['images']
            y = data['joint_values'].float()
            im1,im2,im3, y = im1.to(device), im2.to(device), im3.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(im1, im2, im3).float()
            loss = loss_func(preds, y)
            running_loss +=loss.item()

    val_loss = running_loss/(i+1)

    return(val_loss)