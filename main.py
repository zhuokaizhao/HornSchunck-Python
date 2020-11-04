from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import imageio
import argparse
import numpy as np
import flowiz as fz
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import convolve as filter2

import plot
import load_data
import HornSchunck

def main():

	# input arguments
    parser = argparse.ArgumentParser()
    # input dataset director
    parser.add_argument('--data-dir', action='store', nargs=1, dest='data_dir')
    # output directory (tfrecord in 'data' mode, figure in 'training' mode)
    parser.add_argument('-o', '--output-dir', action='store', nargs=1, dest='output_dir')
    # loss function
    parser.add_argument('-l', '--loss', action='store', nargs=1, dest='loss')
    args = parser.parse_args()

    data_dir = args.data_dir[0]
    figs_dir = args.output_dir[0]
    loss = args.loss[0]

    # read the data
    print(f'\nLoading datasets')
    img1_name_list, img2_name_list, gt_name_list = load_data.read_all(data_dir)
    # load the first sample to determine the image size
    temp = np.asarray(Image.open(img1_name_list[0])) * 1.0 / 255.0
    img_height, img_width = temp.shape

    # define loss
    if loss == 'MSE' or loss == 'RMSE':
        loss_module = torch.nn.MSELoss()
    elif loss == 'MAE':
        loss_module = torch.nn.L1Loss()
    else:
        raise Exception(f'Unrecognized loss function: {loss}')

    # start and end of index (both inclusive)
    start_index = 0
    end_index = 1
    final_size = img_height
    visualize = True

    min_loss = 999
    min_loss_index = 0
    all_losses = []

    for k in range(start_index, end_index+1):
        first_image = np.asarray(Image.open(img1_name_list[k])).reshape(img_height, img_width) * 1.0/255.0
        second_image = np.asarray(Image.open(img2_name_list[k])).reshape(img_height, img_width) * 1.0/255.0
        cur_label_true = fz.read_flow(gt_name_list[k])

        u, v = HornSchunck.HS(first_image, second_image, 1, 100)
        cur_label_pred = np.zeros((img_height, img_width, 2))
        cur_label_pred[:, :, 0] = u
        cur_label_pred[:, :, 1] = v

        cur_loss = loss_module(torch.from_numpy(cur_label_pred), torch.from_numpy(cur_label_true))
        if loss == 'RMSE':
            cur_loss = torch.sqrt(cur_loss)
            # convert to per pixel

        # scale to per pixel
        cur_loss = cur_loss / final_size

        if cur_loss < min_loss:
            min_loss = cur_loss
            min_loss_index = k

        all_losses.append(cur_loss)

        if visualize:
            # visualize the flow
            cur_flow_true = plot.visualize_flow(cur_label_true)
            cur_flow_pred = plot.visualize_flow(cur_label_pred)

            # convert to Image
            cur_flow_true = Image.fromarray(cur_flow_true)
            cur_flow_pred = Image.fromarray(cur_flow_pred)

            # superimpose quiver plot on color-coded images
            # ground truth
            x = np.linspace(0, final_size-1, final_size)
            y = np.linspace(0, final_size-1, final_size)
            y_pos, x_pos = np.meshgrid(x, y)
            skip = 8
            # plt.figure()
            # plt.imshow(cur_flow_true)
            # plt.quiver(y_pos[::skip, ::skip],
            #             x_pos[::skip, ::skip],
            #             cur_label_true[::skip, ::skip, 0],
            #             -cur_label_true[::skip, ::skip, 1])
            # plt.axis('off')
            # true_quiver_path = os.path.join(figs_dir, f'{k}_true.svg')
            # plt.savefig(true_quiver_path, bbox_inches='tight', dpi=1200)
            # print(f'ground truth quiver plot has been saved to {true_quiver_path}')

            # prediction
            plt.figure()
            plt.imshow(cur_flow_pred)
            plt.quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_label_pred[::skip, ::skip, 0],
                        -cur_label_pred[::skip, ::skip, 1])
            plt.axis('off')
            # annotate error
            plt.annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='black', fontsize='large')
            pred_quiver_path = os.path.join(figs_dir, f'HS_{k}_pred.svg')
            plt.savefig(pred_quiver_path, bbox_inches='tight', dpi=1200)
            print(f'prediction quiver plot has been saved to {pred_quiver_path}')

    avg_loss = np.mean(all_losses)
    print(f'\nHonr-Schunck on image [{start_index}:{end_index}] completed\n')
    print(f'Avg loss is {avg_loss}')
    print(f'Min loss is {min_loss} at index {min_loss_index}')

    # save the result to a .text file
    text_path = os.path.join(figs_dir, 'results_sheet.txt')
    np.savetxt(text_path,
                all_losses,
                fmt='%10.5f',
                header=f'{loss} of {end_index-start_index+1} image pairs',
                footer=f'Avg loss is {avg_loss}, Min loss is {min_loss} at index {min_loss_index}')
    print(f'result sheet has been saved at {text_path}')


if __name__ == "__main__":
    main()