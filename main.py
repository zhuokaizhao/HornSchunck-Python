from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
# import imageio
import argparse
import numpy as np
import flowiz as fz
from PIL import Image
from matplotlib import pyplot as plt
# from scipy.ndimage.filters import gaussian_filter
# from scipy.ndimage.filters import convolve as filter2

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

    # start and end of index (both inclusive)
    start_index = 41
    end_index = 41
    final_size = img_height
    visualize = True

    min_loss = 999
    min_loss_index = 0
    all_losses = []

    for k in range(start_index, end_index+1):
        first_image = np.asarray(Image.open(img1_name_list[k])).reshape(img_height, img_width) * 1.0/255.0
        second_image = np.asarray(Image.open(img2_name_list[k])).reshape(img_height, img_width) * 1.0/255.0
        cur_label_true = fz.read_flow(gt_name_list[k]) / 256.0

        u, v = HornSchunck.HS(first_image, second_image, 1, 100)
        cur_label_pred = np.zeros((img_height, img_width, 2))
        cur_label_pred[:, :, 0] = u
        cur_label_pred[:, :, 1] = v

        if loss == 'MSE' or loss == 'RMSE' or loss == 'AEE':
            cur_loss = loss_module(torch.from_numpy(cur_label_pred), torch.from_numpy(cur_label_true))
            if loss == 'RMSE':
                cur_loss = torch.sqrt(cur_loss)
            elif loss == 'AEE':
                sum_endpoint_error = 0
                for i in range(final_size):
                    for j in range(final_size):
                        cur_pred = cur_label_pred[i, j]
                        cur_true = cur_label_true[i, j]
                        cur_endpoint_error = np.linalg.norm(cur_pred-cur_true)
                        sum_endpoint_error += cur_endpoint_error

                # compute the average endpoint error
                aee = sum_endpoint_error / (final_size*final_size)
                # convert to per 100 pixels for comparison purpose
                cur_loss = aee / final_size
        # customized metric that converts into polar coordinates and compare
        elif loss == 'polar':
            # convert both truth and predictions to polar coordinate
            cur_label_true_polar = plot.cart2pol(cur_label_true)
            cur_label_pred_polar = plot.cart2pol(cur_label_pred)
            # absolute magnitude difference and angle difference
            r_diff_mean = np.abs(cur_label_true_polar[:, :, 0]-cur_label_pred_polar[:, :, 0]).mean()
            theta_diff = np.abs(cur_label_true_polar[:, :, 1]-cur_label_pred_polar[:, :, 1])
            # wrap around for angles larger than pi
            theta_diff[theta_diff>2*np.pi] = 2*np.pi - theta_diff[theta_diff>2*np.pi]
            # compute the mean of angle difference
            theta_diff_mean = theta_diff.mean()
            # take the sum as single scalar loss
            cur_loss = r_diff_mean + theta_diff_mean

        if cur_loss < min_loss:
            min_loss = cur_loss
            min_loss_index = k

        all_losses.append(cur_loss)
        print(f't = {k}, RMSE = {cur_loss}')

        if visualize:
            # visualize the flow
            cur_flow_true, max_vel = plot.visualize_flow(cur_label_true)
            print(f'Label max vel magnitude is {max_vel}')
            cur_flow_pred, _ = plot.visualize_flow(cur_label_pred, max_vel=max_vel)
            # print(f'Pred max vel magnitude is {max_vel}')

            # convert to Image
            cur_flow_true = Image.fromarray(cur_flow_true)
            cur_flow_pred = Image.fromarray(cur_flow_pred)

            # superimpose quiver plot on color-coded images
            # ground truth
            x = np.linspace(0, final_size-1, final_size)
            y = np.linspace(0, final_size-1, final_size)
            y_pos, x_pos = np.meshgrid(x, y)
            skip = 8
            plt.figure()
            plt.imshow(cur_flow_true)
            Q = plt.quiver(y_pos[::skip, ::skip],
                            x_pos[::skip, ::skip],
                            cur_label_true[::skip, ::skip, 0]/max_vel,
                            -cur_label_true[::skip, ::skip, 1]/max_vel,
                            scale=4.0,
                            scale_units='inches')
            Q._init()
            assert isinstance(Q.scale, float)
            plt.axis('off')
            true_quiver_path = os.path.join(figs_dir, f'{k}_true.svg')
            plt.savefig(true_quiver_path, bbox_inches='tight', dpi=1200)
            print(f'ground truth quiver plot has been saved to {true_quiver_path}')

            # prediction
            plt.figure()
            plt.imshow(cur_flow_pred)
            plt.quiver(y_pos[::skip, ::skip],
                        x_pos[::skip, ::skip],
                        cur_label_pred[::skip, ::skip, 0]/max_vel,
                        -cur_label_pred[::skip, ::skip, 1]/max_vel,
                        scale=Q.scale,
                        scale_units='inches')
            plt.axis('off')
            # annotate error
            if loss == 'polar':
                        plt.annotate(f'Magnitude MAE: ' + '{:.3f}'.format(r_diff_mean), (5, 10), color='white', fontsize='medium')
                        plt.annotate(f'Angle MAE: ' + '{:.3f}'.format(theta_diff_mean), (5, 20), color='white', fontsize='medium')
            else:
                plt.annotate(f'{loss}: ' + '{:.3f}'.format(cur_loss), (5, 10), color='white', fontsize='large')
            pred_quiver_path = os.path.join(figs_dir, f'HS_{k}_pred.svg')
            plt.savefig(pred_quiver_path, bbox_inches='tight', dpi=1200)
            print(f'prediction quiver plot has been saved to {pred_quiver_path}')

            # plot error difference
            pred_error = np.sqrt((cur_label_pred[:,:,0]-cur_label_true[:,:,0])**2 \
                                        + (cur_label_pred[:,:,1]-cur_label_true[:,:,1])**2)
            plt.figure()
            plt.imshow(pred_error, cmap='PuBuGn', interpolation='nearest', vmin=0.0,  vmax=1.0)
            error_path = os.path.join(figs_dir, f'hs_{k}_error.svg')
            plt.axis('off')
            cbar = plt.colorbar()
            # cbar.set_label('Endpoint error')
            plt.savefig(error_path, bbox_inches='tight', dpi=1200)
            print(f'error magnitude plot has been saved to {error_path}')

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