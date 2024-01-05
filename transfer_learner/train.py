from source.robot_dataset import RobotImageDataset
from source.model import RobotNet
from utils.trainers import train_one_epoch, validate_one_epoch
import wandb
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim



def main():
    run = wandb.init()

    # Params
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 32
    EPOCH = 3
    IN_DIM1 = 256
    IN_DIM2 = 16


    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Data
    robotdata = RobotImageDataset('./source/data/UR5_positions.csv', './source/data/images/', multi_input=True)
    print('Set up RobotImageClass...')
    traindata, validdata = random_split(robotdata, [0.8,0.2])
    trainloader = DataLoader(traindata, batch_size=BATCH_SIZE, shuffle=True)
    validloader = DataLoader(validdata, batch_size=BATCH_SIZE, shuffle=True)
    print('Set up dataloaders...')
    multi_model = RobotNet(IN_DIM1, IN_DIM2, n_inputs = 3)
    print('Set up model...')


    # Initialize
    multi_model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(multi_model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1,EPOCH):
        
        print('Beginning training loop...')
        tr_loss = train_one_epoch(trainloader, multi_model, criterion, optimizer, device)
        print('Beginning validation loop...')
        val_loss = validate_one_epoch(validloader, multi_model, criterion, optimizer, device)
        print(f'Epoch {epoch}, train loss: {tr_loss}, val loss: {val_loss}')
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": val_loss,
            }
        )
    run.finish()

if __name__ == "__main__":
    main()