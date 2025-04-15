import torch
import os
import numpy as np
import torch.utils.data as data
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from networks import GMM, load_checkpoint
from cp_dataset import CPDataset, CPDataLoader
import argparse


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="test_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=100)
    parser.add_argument("--keep_step", type=int, default=100)  # Update here
    parser.add_argument("--decay_step", type=int, default=100)  # Update here
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

# Load the model and checkpoint
def load_model(opt):
    model = GMM(opt)
    if opt.checkpoint and os.path.exists(opt.checkpoint):
        load_checkpoint(model, opt.checkpoint)
    model.eval()
    return model

# Prepare the data
def prepare_data(opt):
    test_dataset = CPDataset(opt)
    test_loader = CPDataLoader(opt, test_dataset)
    return test_loader

def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            agnostic = batch['agnostic'].to(device)
            c = batch['cloth'].to(device)
            im_c = batch['parse_cloth'].to(device)

            grid, _ = model(agnostic, c)
            warped_cloth = torch.nn.functional.grid_sample(c, grid, padding_mode='border')

            # Convert tensors to numpy arrays for PSNR and SSIM
            warped_cloth_np = warped_cloth.cpu().numpy().transpose(0, 2, 3, 1)
            im_c_np = im_c.cpu().numpy().transpose(0, 2, 3, 1)

            for i in range(warped_cloth_np.shape[0]):
                warped_cloth_img = np.clip(warped_cloth_np[i], 0, 1)
                im_c_img = np.clip(im_c_np[i], 0, 1)

                psnr_value = psnr(im_c_img, warped_cloth_img, data_range=1)

                # Adjust win_size or specify explicitly
                try:
                    ssim_value = ssim(im_c_img, warped_cloth_img, multichannel=True, data_range=1, win_size=3)
                except ValueError as e:
                    print(f"SSIM error: {e}")
                    ssim_value = 0.0

                total_psnr += psnr_value
                total_ssim += ssim_value
                count += 1

    average_psnr = total_psnr / count
    average_ssim = total_ssim / count

    print(f"Average PSNR: {average_psnr:.4f}")
    print(f"Average SSIM: {average_ssim:.4f}")


class CPDataLoader(data.DataLoader):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__(dataset, batch_size=opt.batch_size, shuffle=opt.shuffle,
                                           num_workers=opt.workers, pin_memory=True)

    def __iter__(self):
        return super(CPDataLoader, self).__iter__()

    def __next__(self):
        return super(CPDataLoader, self).__next__()
    
# Main function
def main():
    opt = get_opt()
    print(f"Evaluating stage: {opt.stage}, named: {opt.name}")
    
    # Determine the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = load_model(opt)
    
    # Prepare data
    test_loader = prepare_data(opt)
    
    # Evaluate the model
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
