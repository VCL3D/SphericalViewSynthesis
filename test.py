import argparse
import os
import sys
import numpy

import torch
import torchvision

import models
import dataset
import utils
from filesystem import file_utils

import supervision as L
import exporters as IO
import spherical as S360

def parse_arguments(args):
    usage_text = (
        "Semi-supervised Spherical Depth Estimation Testing."        
    )
    parser = argparse.ArgumentParser(description=usage_text)
    # enumerables
    parser.add_argument('-b',"--batch_size", type=int, help="Test a <batch_size> number of samples each iteration.")    
    parser.add_argument('--save_iters', type=int, default=100, help='Maximum test iterations whose results will be saved.')
    # paths    
    parser.add_argument("--test_path", type=str, help="Path to the testing file containing the test set file paths")
    parser.add_argument("--save_path", type=str, help="Path to the folder where the models and results will be saved at.")
    # model
    parser.add_argument("--configuration", required=False, type=str, default='mono', help="Data loader configuration <mono>, <lr>, <ud>, <tc>", choices=['mono', 'lr', 'ud', 'tc'])
    parser.add_argument('--weights', type=str, help='Path to the trained weights file.')
    parser.add_argument('--model', default="default", type=str, help='Model selection argument.')    
    # hardware
    parser.add_argument('-g','--gpu', type=str, default='0', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    # other
    parser.add_argument('-n','--name', type=str, default='default_name', help='The name of this train/test. Used when storing information.')    
    parser.add_argument("--visdom", type=str, nargs='?', default=None, const="127.0.0.1", help="Visdom server IP (port defaults to 8097)")
    parser.add_argument("--visdom_iters", type=int, default=400, help = "Iteration interval that results will be reported at the visdom server for visualization.")
    # metrics
    parser.add_argument("--depth_thres", type=float, default=20.0, help = "Depth threshold - depth clipping.")
    parser.add_argument("--width", type=float, default=512, help = "Spherical image width.")
    parser.add_argument("--baseline", type=float, default=0.26, help = "Stereo baseline distance (in either axis).")
    parser.add_argument("--median_scale", required=False, default=False, action="store_true", help = "Perform median scaling before calculating metrics.")
    parser.add_argument("--spherical_weights", required=False, default=False, action="store_true", help = "Use spherical weighting when calculating the metrics.")
    parser.add_argument("--spherical_sampling", required=False, default=False, action="store_true", help = "Use spherical sampling when calculating the metrics.")
    # save options    
    parser.add_argument("--save_recon", required=False, default=False, action="store_true", help = "Flag to toggle reconstructed result saving.")
    parser.add_argument("--save_original", required=False, default=False, action="store_true", help = "Flag to toggle input (image) saving.")
    parser.add_argument("--save_depth", required=False, default=False, action="store_true", help = "Flag to toggle output (depth) saving.")    
    return parser.parse_known_args(args)

def compute_errors(gt, pred, invalid_mask, weights, sampling, mode='cpu', median_scale=False):
    b, _, __, ___ = gt.size()
    scale = torch.median(gt.reshape(b, -1), dim=1)[0] / torch.median(pred.reshape(b, -1), dim=1)[0]\
        if median_scale else torch.tensor(1.0).expand(b, 1, 1, 1).to(gt.device) 
    pred = pred * scale.reshape(b, 1, 1, 1)
    valid_sum = torch.sum(~invalid_mask, dim=[1, 2, 3], keepdim=True)
    gt[invalid_mask] = 0.0
    pred[invalid_mask] = 0.0
    thresh = torch.max((gt / pred), (pred / gt))
    thresh[invalid_mask | (sampling < 0.5)] = 2.0
    
    sum_dims = [1, 2, 3]
    delta_valid_sum = torch.sum(~invalid_mask & (sampling > 0), dim=[1, 2, 3], keepdim=True)
    delta1 = (thresh < 1.25   ).float().sum(dim=sum_dims, keepdim=True).float() / delta_valid_sum.float()
    delta2 = (thresh < (1.25 ** 2)).float().sum(dim=sum_dims, keepdim=True).float() / delta_valid_sum.float()
    delta3 = (thresh < (1.25 ** 3)).float().sum(dim=sum_dims, keepdim=True).float() / delta_valid_sum.float()

    rmse = (gt - pred) ** 2
    rmse[invalid_mask] = 0.0
    rmse_w = rmse * weights
    rmse_mean = torch.sqrt(rmse_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log[invalid_mask] = 0.0
    rmse_log_w = rmse_log * weights
    rmse_log_mean = torch.sqrt(rmse_log_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float())

    abs_rel = (torch.abs(gt - pred) / gt)
    abs_rel[invalid_mask] = 0.0
    abs_rel_w = abs_rel * weights
    abs_rel_mean = abs_rel_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float()

    sq_rel = (((gt - pred)**2) / gt)
    sq_rel[invalid_mask] = 0.0
    sq_rel_w = sq_rel * weights
    sq_rel_mean = sq_rel_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float()

    return (abs_rel_mean, abs_rel), (sq_rel_mean, sq_rel), (rmse_mean, rmse), \
        (rmse_log_mean, rmse_log), delta1, delta2, delta3

def spiral_sampling(grid, percentage):
    b, c, h, w = grid.size()    
    N = torch.tensor(h*w*percentage).int().float()    
    sampling = torch.zeros_like(grid)[:, 0, :, :].unsqueeze(1)
    phi_k = torch.tensor(0.0).float()
    for k in torch.arange(N - 1):
        k = k.float() + 1.0
        h_k = -1 + 2 * (k - 1) / (N - 1)
        theta_k = torch.acos(h_k)
        phi_k = phi_k + torch.tensor(3.6).float() / torch.sqrt(N) / torch.sqrt(1 - h_k * h_k) \
            if k > 1.0 else torch.tensor(0.0).float()
        phi_k = torch.fmod(phi_k, 2 * numpy.pi)
        sampling[:, :, int(theta_k / numpy.pi * h) - 1, int(phi_k / numpy.pi / 2 * w) - 1] += 1.0
    return (sampling > 0).float()

if __name__ == "__main__":
    args, unknown = parse_arguments(sys.argv)
    gpus = [int(id) for id in args.gpu.split(',') if int(id) >= 0]
    # device & visualizers
    device = torch.device("cuda:{}" .format(gpus[0])\
        if torch.cuda.is_available() and len(gpus) > 0 and gpus[0] >= 0\
        else "cpu")
    plot_visualizer, image_visualizer = (utils.NullVisualizer(), utils.NullVisualizer())\
        if args.visdom is None\
        else (
            utils.VisdomPlotVisualizer(args.name + "_test_plots_", args.visdom),
            utils.VisdomImageVisualizer(args.name + "_test_images_", args.visdom,\
                count=2 if 2 <= args.batch_size else args.batch_size)
        )
    image_visualizer.update_epoch(0)
    # model
    model_params = { 'width': 512, 'configuration': args.configuration }
    model = models.get_model(args.model, model_params)
    utils.init.initialize_weights(model, args.weights, pred_bias=None)
    if (len(gpus) > 1):  
        model = torch.nn.parallel.DataParallel(model, gpus)
    model = model.to(device)
    # test data
    width, height = args.width, args.width // 2
    test_data = dataset.dataset_360D.Dataset360D(args.test_path, " ", args.configuration, [height, width])
    test_data_iterator = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,\
        num_workers=args.batch_size // 4 // (2 if len(gpus) > 0 else 1), pin_memory=False, shuffle=False)
    fs = file_utils.Filesystem()
    fs.mkdir(args.save_path)
    print("Test size : {}".format(args.batch_size * test_data_iterator.__len__()))    
    # params & error vars
    max_save_iters = args.save_iters if args.save_iters > 0\
        else args.batch_size * test_data_iterator.__len__()
    errors = numpy.zeros((7, args.batch_size * test_data_iterator.__len__()), numpy.float32)    
    weights = S360.weights.theta_confidence(
        S360.grid.create_spherical_grid(width)
    ).to(device) if args.spherical_weights else torch.ones(1, 1, height, width).to(device)
    sampling = spiral_sampling(S360.grid.create_image_grid(width, height), 0.25).to(device) \
        if args.spherical_sampling else torch.ones(1, 1, height, width).to(device)
    # loop over test set
    model.eval()
    with torch.no_grad():            
        counter = 0
        uvgrid = S360.grid.create_image_grid(width, height).to(device)
        sgrid = S360.grid.create_spherical_grid(width).to(device)
        for test_batch_id , test_batch in enumerate(test_data_iterator):
            ''' Data '''
            left_rgb = test_batch['leftRGB'].to(device)            
            left_depth = test_batch['leftDepth'].to(device)
            if 'rightRGB' in test_batch:
                right_rgb = test_batch['rightRGB'].to(device)            
            mask = (left_depth > args.depth_thres)            
            b, c, h, w = left_rgb.size()            
            ''' Prediction '''
            left_depth_pred = torch.abs(model(left_rgb))
            ''' Errors '''
            abs_rel_t, sq_rel_t, rmse_t, rmse_log_t, delta1, delta2, delta3\
                = compute_errors(left_depth, left_depth_pred, mask, weights=weights, sampling=sampling, \
                    mode='gpu' if torch.cuda.is_available() and len(gpus) > 0 and gpus[0] >= 0 else "cpu", \
                    median_scale=args.median_scale)
            ''' Visualize & Append Errors '''
            for i in range(b):
                idx = counter + i
                errors[:, idx] = abs_rel_t[0][i], sq_rel_t[0][i], rmse_t[0][i], \
                    rmse_log_t[0][i], delta1[i], delta2[i], delta3[i]
                for j in range(7):
                    plot_visualizer.append_loss(1, idx, torch.tensor(errors[0, idx]), "abs_rel")
                    plot_visualizer.append_loss(1, idx, torch.tensor(errors[1, idx]), "sq_rel")
                    plot_visualizer.append_loss(1, idx, torch.tensor(errors[2, idx]), "rmse")
                    plot_visualizer.append_loss(1, idx, torch.tensor(errors[3, idx]), "rmse_log")
                    plot_visualizer.append_loss(1, idx, torch.tensor(errors[4, idx]), "delta1")
                    plot_visualizer.append_loss(1, idx, torch.tensor(errors[5, idx]), "delta2")
                    plot_visualizer.append_loss(1, idx, torch.tensor(errors[6, idx]), "delta3")            
            ''' Store '''
            if counter < args.save_iters:
                if args.save_original:
                    IO.image.save_image(os.path.join(args.save_path,\
                        str(counter) + "_" + args.name + "_#_left.png"), left_rgb)
                if args.save_depth:
                    IO.image.save_data(os.path.join(args.save_path,\
                        str(counter) + "_" + args.name + "_#_depth.exr"), left_depth_pred, scale=1.0)
                if args.save_recon:
                    rads = sgrid.expand(b, -1, -1, -1)
                    uv = uvgrid.expand(b, -1, -1, -1)
                    disp = torch.cat(
                        (
                            S360.derivatives.dphi_horizontal(rads, left_depth_pred, args.baseline),
                            S360.derivatives.dtheta_horizontal(rads, left_depth_pred, args.baseline)
                        ), dim=1
                    )
                    right_render_coords = uv + disp
                    right_render_coords[:, 0, :, :] = torch.fmod(right_render_coords[:, 0, :, :] + width, width)
                    right_render_coords[torch.isnan(right_render_coords)] = 0.0
                    right_render_coords[torch.isinf(right_render_coords)] = 0.0
                    right_rgb_t, right_mask_t = L.splatting.render(left_rgb, left_depth_pred, right_render_coords, max_depth=args.depth_thres)
                    IO.image.save_image(os.path.join(args.save_path,\
                        str(counter) + "_" + args.name + "_#_right_t.png"), right_rgb_t) 
            counter += b
            ''' Visualize Predictions '''
            if args.visdom_iters > 0 and (counter + 1) % args.visdom_iters <= args.batch_size:                
                image_visualizer.show_separate_images(left_rgb, 'input')
                if 'rightRGB' in test_batch:
                    image_visualizer.show_separate_images(right_rgb, 'target')
                image_visualizer.show_map(left_depth_pred, 'depth')
                if args.save_recon:
                    image_visualizer.show_separate_images(right_rgb_t, 'recon')
        mean_errors = errors.mean(1)
        error_names = ['abs_rel','sq_rel','rmse','log_rmse','delta1','delta2','delta3']
        print("Results ({}): ".format(args.name))
        print("\t{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
        print("\t{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors))        
        
            
