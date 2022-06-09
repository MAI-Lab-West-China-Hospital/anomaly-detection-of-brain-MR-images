import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism
from monai.visualize import plot_2d_or_3d_image
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    AddChannelD,
    Compose,
    LoadImageD,
    ScaleIntensityD,
    EnsureTypeD,
)

from model import Encoder, Decoder

from glob import glob
import random
import os
import time


def form_results(imgsize, z_dim):
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    date = time.strftime('%Y%m%d', time.localtime(time.time()))
    results_path = '../Results'
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    folder_name = "/{0}_insize{1}_z{2}_denseAE_sigmoid".format(date, imgsize, z_dim)
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path


def train(trainloader, valloader, h, w, z, device_ids, epochs):
    encoder = Encoder(h, w, z, z_dim=z_dim)
    decoder = Decoder(h, w, z, z_dim=z_dim)

    # if distribution
    encoder = nn.DataParallel(encoder, device_ids=device_ids).to(device)
    decoder = nn.DataParallel(decoder, device_ids=device_ids).to(device)

    ae_loss = nn.MSELoss()

    optimizer_ae = optim.Adam([{'params': encoder.parameters()},
                               {'params': decoder.parameters()}], lr=1e-4, weight_decay=1e-5)

    tensorboard_path, saved_model_path, log_path = form_results(f'{h}-{w}-{z}', z_dim)
    writer = SummaryWriter(tensorboard_path)

    step = 0
    best_loss = 100
    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        encoder.train()
        decoder.train()

        autoencoder_loss_epoch = 0.0

        for data in trainloader:
            img = data['im'].to(device)
            # ==========forward=========
            z = encoder(img)
            x_hat = decoder(z)

            # ==========compute the loss and backpropagate=========
            encoder_decoder_loss = ae_loss(x_hat, img)

            optimizer_ae.zero_grad()
            encoder_decoder_loss.backward()
            optimizer_ae.step()

            # ========METRICS===========
            autoencoder_loss_epoch += encoder_decoder_loss.item()

            writer.add_scalar('step_train_loss', encoder_decoder_loss, step)

            step += 1

        train_loss = autoencoder_loss_epoch / len(trainloader)
        val_loss = val(valloader, encoder, decoder)

        print('train_loss: {:.4f}'.format(train_loss))
        print('val_loss: {:.4f}'.format(val_loss))
        writer.add_scalars('train and val loss per epoch', {'train_loss': train_loss,
                                                            'val_loss': val_loss
                                                            }, epoch + 1)
        plot_2d_or_3d_image(img, epoch + 1, writer, index=0, frame_dim=-1, tag='image')
        plot_2d_or_3d_image(x_hat, epoch + 1, writer, index=0, frame_dim=-1, tag='recon image')

        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            torch.save({
                'epoch': epoch + 1,
                'encoder': encoder.state_dict(),
            }, saved_model_path + f'/encoder_{epoch + 1}.pth')

            torch.save({
                'epoch': epoch + 1,
                'decoder': decoder.state_dict(),
            }, saved_model_path + f'/decoder_{epoch + 1}.pth')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'encoder': encoder.state_dict(),
            }, saved_model_path + f'/encoder_best.pth')

            torch.save({
                'epoch': epoch + 1,
                'decoder': decoder.state_dict(),
            }, saved_model_path + f'/decoder_best.pth')
            print(f'saved best model in epoch: {epoch+1}')
    writer.close()


def val(dataloader, encoder, decoder):
    encoder.eval()
    decoder.eval()

    ae_loss = nn.MSELoss()
    autoencoder_loss = 0.0

    with torch.no_grad():
        for data in dataloader:
            img = data['im'].to(device)
            # ==========forward=========
            z = encoder(img)
            x_hat = decoder(z)

            # ==========compute the loss=========
            encoder_decoder_loss = ae_loss(x_hat, img)
            autoencoder_loss += encoder_decoder_loss.item()

        tol_loss = autoencoder_loss / len(dataloader)

    return tol_loss


if __name__ == '__main__':
    set_determinism(seed=42)
    device_ids = [0, 1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========== MONAI Dataloader =========
    IXIT2_data_dir = 'data/IXI_Brats_t2/register1mm/IXIT2'
    IXIT2_filenames = glob(IXIT2_data_dir + '/*.nii.gz')
    random.shuffle(IXIT2_filenames)

    # Split into training and testing
    test_frac = 0.2
    num_ims = len(IXIT2_filenames)
    num_test = int(num_ims * test_frac)
    num_train = num_ims - num_test
    train_datadict = [{"im": fname} for fname in IXIT2_filenames[:num_train]]
    val_datadict = [{"im": fname} for fname in IXIT2_filenames[-num_test:]]
    print(f"total number of images: {num_ims}")
    print(f"number of images for training: {len(train_datadict)}")
    print(f"number of images for testing: {len(val_datadict)}")

    batch_size = 8
    num_workers = 12
    z_dim = 512
    epochs = 200

    transforms = Compose(
        [
            LoadImageD(keys=["im"]),
            AddChannelD(keys=["im"]),
            ScaleIntensityD(keys=["im"]),
            EnsureTypeD(keys=["im"]),
        ]
    )

    train_ds = CacheDataset(train_datadict, transforms, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ds = CacheDataset(val_datadict, transforms, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train(train_loader, val_loader, 160, 192, 96, device_ids, epochs)
