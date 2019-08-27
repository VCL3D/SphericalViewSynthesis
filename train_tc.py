import argparse
import os
import sys
import numpy

import torch

import models
import dataset
import utils

import supervision as L
import exporters as IO
import spherical as S360

def parse_arguments(args):
    usage_text = (
        "Omnidirectional Trinocular Stereo Placement (Up-Down & Left-Right , UD+LR) Training"
    )
    parser = argparse.ArgumentParser(description=usage_text)
   # durations
    parser.add_argument('-e',"--epochs", type=int, help="Train for a total number of <epochs> epochs.")
    parser.add_argument('-b',"--batch_size", type=int, help="Train with a <batch_size> number of samples each train iteration.")
    parser.add_argument("--test_batch_size", default=1, type=int, help="Test with a <batch_size> number of samples each test iteration.")    
    parser.add_argument('-d','--disp_iters', type=int, default=50, help='Log training progress (i.e. loss etc.) on console every <disp_iters> iterations.')
    parser.add_argument('--save_iters', type=int, default=100, help='Maximum test iterations to perform each test run.')
    # paths
    parser.add_argument("--configuration", required=False, type=str, default='mono', help="Data loader configuration <mono>, <lr>, <ud>, <tc>", choices=['mono', 'lr', 'ud', 'tc'])
    parser.add_argument("--test_path", type=str, help="Path to the testing file containing the test set file paths")
    parser.add_argument("--save_path", type=str, help="Path to the folder where the models and results will be saved at.")
    # model
    parser.add_argument("--configuration", required = False, type = str, default='tc', help = "Training configuration <mono>, <lr>, <ud>, <tc>", choices=['mono', 'lr', 'ud', 'tc'])
    parser.add_argument('--weight_init', type=str, default="xavier", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    parser.add_argument('--model', default="default", type=str, help='Model selection argument.')    
    # optimization
    parser.add_argument('-o','--optimizer', type=str, default="adam", help='The optimizer that will be used during training.')
    parser.add_argument("--opt_state", type=str, help="Path to stored optimizer state file to continue training)")
    parser.add_argument('-l','--lr', type=float, default=0.0002, help='Optimization Learning Rate.')
    parser.add_argument('-m','--momentum', type=float, default=0.9, help='Optimization Momentum.')
    parser.add_argument('--momentum2', type=float, default=0.999, help='Optimization Second Momentum (optional, only used by some optimizers).')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Optimization Epsilon (optional, only used by some optimizers).')
    parser.add_argument('--weight_decay', type=float, default=0, help='Optimization Weight Decay.')    
    # hardware
    parser.add_argument('-g','--gpu', type=str, default='0', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    # other
    parser.add_argument('-n','--name', type=str, default='default_name', help='The name of this train/test. Used when storing information.')    
    parser.add_argument("--visdom", type=str, nargs='?', default=None, const="127.0.0.1", help="Visdom server IP (port defaults to 8097)")
    parser.add_argument("--visdom_iters", type=int, default=400, help = "Iteration interval that results will be reported at the visdom server for visualization.")
    parser.add_argument("--seed", type=int, default=1337, help="Fixed manual seed, zero means no seeding.")
    # network specific params
    parser.add_argument("--photo_w", type=float, default=1.0, help = "Photometric loss weight.")
    parser.add_argument("--photo_ratio", type=float, default=0.5, help = "Ratio between right (1-ratio) and up (ratio) photometric loss.")
    parser.add_argument("--smooth_reg_w", type=float, default=0.1, help = "Smoothness regularization weight.")    
    parser.add_argument("--ssim_window", type=int, default=7, help = "Kernel size to use in SSIM calculation.")
    parser.add_argument("--ssim_mode", type=str, default='gaussian', help = "Type of SSIM averaging (either gaussian or box).")
    parser.add_argument("--ssim_std", type=float, default=1.5, help = "SSIM standard deviation value used when creating the gaussian averaging kernels.")
    parser.add_argument("--ssim_alpha", type=float, default=0.85, help = "Alpha factor to weight the SSIM and L1 losses, where a x SSIM and (1 - a) x L1.")
    parser.add_argument("--pred_bias", type=float, default=5.0, help = "Initialize prediction layers' bias to the given value (helps convergence).")
    # details
    parser.add_argument("--depth_thres", type=float, default=20.0, help = "Depth threshold - depth clipping.")
    parser.add_argument("--baseline", type=float, default=0.26, help = "Stereo baseline distance (in either axis).")
    parser.add_argument("--width", type=float, default=512, help = "Spherical image width.")
    return parser.parse_known_args(args)

if __name__ == "__main__":
    args, unknown = parse_arguments(sys.argv)
    gpus = [int(id) for id in args.gpu.split(',') if int(id) >= 0]
    # device & visualizers
    device, visualizers, model_params = utils.initialize(args)
    plot_viz = visualizers[0]
    img_viz = visualizers[1]
    # model
    model = models.get_model(args.model, model_params)    
    utils.init.initialize_weights(model, args.weight_init, pred_bias=args.pred_bias)
    if (len(gpus) > 1):        
        model = torch.nn.parallel.DataParallel(model, gpus)
    model = model.to(device)
    # optimizer
    optimizer = utils.init_optimizer(model, args)
    # train data
    train_data = dataset.dataset_360D.Dataset360D(args.train_path, " ", args.configuration, [256, 512])
    train_data_iterator = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,\
        num_workers=args.batch_size // len(gpus) // 4, pin_memory=False, shuffle=True)
    # test data
    test_data = dataset.dataset_360D.Dataset360D(args.test_path, " ", args.configuration, [256, 512])
    test_data_iterator = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size,\
        num_workers=args.batch_size // len(gpus) // 4, pin_memory=False, shuffle=True)
    print("Data size : {0} | Test size : {1}".format(\
        args.batch_size * train_data_iterator.__len__(), \
        args.test_batch_size * test_data_iterator.__len__()))    
    # params
    width = args.width
    height = args.width // 2    
    photo_params = L.photometric.PhotometricLossParameters(
        alpha=args.ssim_alpha, l1_estimator='none', ssim_estimator='none',
        ssim_mode=args.ssim_mode, std=args.ssim_std, window=args.ssim_window
    )
    iteration_counter = 0
    # meters
    total_loss = utils.AverageMeter()
    running_photo_loss_lr = utils.AverageMeter()
    running_photo_loss_ud = utils.AverageMeter()
    running_depth_smooth_loss = utils.AverageMeter()
    # train / test loop
    model.train()
    plot_viz.config(**vars(args))
    for epoch in range(args.epochs):
        print("Training | Epoch: {}".format(epoch))
        img_viz.update_epoch(epoch)
        for batch_id, batch in enumerate(train_data_iterator):
            optimizer.zero_grad()
            active_loss = torch.tensor(0.0).to(device)
            ''' Data '''
            left_rgb = batch['leftRGB'].to(device)
            b, _, __, ___ = left_rgb.size()
            expand_size = (b, -1, -1, -1)
            sgrid = S360.grid.create_spherical_grid(width).to(device)
            uvgrid = S360.grid.create_image_grid(width, height).to(device)            
            right_rgb = batch['rightRGB'].to(device)
            up_rgb = batch['upRGB'].to(device)
            left_depth = batch['leftDepth'].to(device)
            up_depth = batch['upDepth'].to(device)
            right_depth = batch['rightDepth'].to(device)
            ''' Prediction '''
            left_depth_pred = torch.abs(model(left_rgb))
            ''' Forward Rendering LR '''
            disp_lr = torch.cat(
                (
                    S360.derivatives.dphi_horizontal(sgrid, left_depth_pred, args.baseline),
                    S360.derivatives.dtheta_horizontal(sgrid, left_depth_pred, args.baseline)
                ),
                dim=1
            )            
            right_render_coords = uvgrid + disp_lr
            right_render_coords[:, 0, :, :] = torch.fmod(right_render_coords[:, 0, :, :] + width, width)
            right_render_coords[torch.isnan(right_render_coords)] = 0.0
            right_render_coords[torch.isinf(right_render_coords)] = 0.0            
            right_rgb_t, right_mask_t = L.splatting.render(left_rgb, left_depth_pred,\
                right_render_coords, max_depth=args.depth_thres)
            ''' Forward Rendering UD '''
            disp_ud = torch.cat(
                (
                    torch.zeros_like(left_depth_pred),
                    S360.derivatives.dtheta_vertical(sgrid, left_depth_pred, args.baseline)
                ),
                dim=1
            )
            up_render_coords = uvgrid + disp_ud
            up_render_coords[torch.isnan(up_render_coords)] = 0.0
            up_render_coords[torch.isinf(up_render_coords)] = 0.0            
            up_rgb_t, up_mask_t = L.splatting.render(left_rgb, left_depth_pred,\
                up_render_coords, max_depth=args.depth_thres)
            ''' Loss LR '''
            right_cutoff_mask = (right_depth < args.depth_thres)
            attention_weights_lr = S360.weights.phi_confidence(
                S360.grid.create_spherical_grid(width)).to(device)
            # attention_weights_lr = S360.weights.spherical_confidence(              
            #     S360.grid.create_spherical_grid(width), zero_low=0.001
            # ).to(device)
            photo_loss_lr = L.photometric.calculate_loss(right_rgb_t, right_rgb, photo_params,                
               mask=right_cutoff_mask, weights=attention_weights_lr)
            active_loss += photo_loss_lr * args.photo_w * (1 - args.photo_ratio)
            ''' Loss UD '''
            up_cutoff_mask = (up_depth < args.depth_thres)
            attention_weights_ud = S360.weights.theta_confidence(                
                S360.grid.create_spherical_grid(width)).to(device)
            photo_loss_ud = L.photometric.calculate_loss(up_rgb_t, up_rgb, photo_params,                
               mask=up_cutoff_mask, weights=attention_weights_ud)
            active_loss += photo_loss_ud * args.photo_w * args.photo_ratio
            ''' Loss Prior (3D Smoothness) '''
            left_xyz = S360.cartesian.coords_3d(sgrid, left_depth_pred)
            dI_dxyz = S360.derivatives.dV_dxyz(left_xyz)
            tc_cuttof_mask = right_cutoff_mask & up_cutoff_mask                          
            guidance_duv = S360.derivatives.dI_duv(left_rgb)                
            depth_smooth_loss = L.smoothness.guided_smoothness_loss(
                dI_dxyz, guidance_duv, tc_cuttof_mask, (1.0 - attention_weights_ud)
                * tc_cuttof_mask.type(attention_weights_ud.dtype)
            )            
            active_loss += depth_smooth_loss * args.smooth_reg_w
            ''' Update Params '''            
            active_loss.backward()
            optimizer.step()
            ''' Visualize'''
            total_loss.update(active_loss)
            running_depth_smooth_loss.update(depth_smooth_loss)
            running_photo_loss_lr.update(photo_loss_lr)
            running_photo_loss_ud.update(photo_loss_ud)
            iteration_counter += b           
            if (iteration_counter + 1) % args.disp_iters <= args.batch_size:
                print("Epoch: {}, iteration: {}\nPhotometric (LR-UD): {} - {}\nSmoothness: {}\nTotal average loss: {}\n"\
                    .format(epoch, iteration_counter, running_photo_loss_lr.avg, \
                         running_photo_loss_ud.avg, running_depth_smooth_loss.avg, total_loss.avg))
                plot_viz.append_loss(epoch + 1, iteration_counter, total_loss.avg, "avg")
                plot_viz.append_loss(epoch + 1, iteration_counter, running_photo_loss_lr.avg, "photo_lr")
                plot_viz.append_loss(epoch + 1, iteration_counter, running_photo_loss_ud.avg, "photo_ud")
                plot_viz.append_loss(epoch + 1, iteration_counter, running_depth_smooth_loss.avg, "smooth")
                total_loss.reset()
                running_photo_loss_lr.reset()
                running_photo_loss_ud.reset()
                running_depth_smooth_loss.reset()            
            if args.visdom_iters > 0 and (iteration_counter + 1) % args.visdom_iters <= args.batch_size:
                img_viz.show_separate_images(left_rgb, 'input')
                img_viz.show_separate_images(right_rgb, 'right')
                img_viz.show_separate_images(up_rgb, 'up')
                img_viz.show_map(left_depth_pred, 'depth')
                img_viz.show_separate_images(torch.clamp(right_rgb_t, min=0.0, max=1.0), 'recon_lr')
                img_viz.show_separate_images(torch.clamp(up_rgb_t, min=0.0, max=1.0), 'recon_ud')
        ''' Save '''
        print("Saving model @ epoch #" + str(epoch))
        utils.checkpoint.save_network_state(model, optimizer, epoch,\
            args.name + "_model_state", args.save_path)
        ''' Test '''
        print("Testing model @ epoch #" + str(epoch))
        model.eval()
        with torch.no_grad():
            rmse_avg = torch.tensor(0.0).float()
            counter = torch.tensor(0.0).float()
            for test_batch_id , test_batch in enumerate(test_data_iterator):
                left_rgb = test_batch['leftRGB'].to(device)
                b, c, h, w = left_rgb.size()
                rads = sgrid.expand(b, -1, -1, -1)
                uv = uvgrid.expand(b, -1, -1, -1)
                left_depth_pred = torch.abs(model(left_rgb))
                left_depth = test_batch['leftDepth'].to(device)
                left_depth[torch.isnan(left_depth)] = 50.0
                left_depth[torch.isinf(left_depth)] = 50.0
                mse = (left_depth_pred ** 2) - (left_depth ** 2)
                mse[torch.isnan(mse)] = 0.0
                mse[torch.isinf(mse)] = 0.0
                mask = (left_depth < args.depth_thres).float()
                if torch.sum(mask) == 0:
                    continue
                rmse = torch.sqrt(torch.sum(mse * mask) / torch.sum(mask).float())
                if not torch.isnan(rmse):
                    rmse_avg += rmse.cpu().float()
                    counter += torch.tensor(b).float()
                if counter < args.save_iters:
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
                    # save
                    IO.image.save_image(os.path.join(args.save_path,\
                        str(epoch) + "_" + str(counter) + "_#_left.png"), left_rgb)
                    IO.image.save_image(os.path.join(args.save_path,\
                        str(epoch) + "_" + str(counter) + "_#_right_t.png"), right_rgb_t)
                    IO.image.save_data(os.path.join(args.save_path,\
                        str(epoch) + "_" + str(counter) + "_#_depth.exr"), left_depth_pred, scale=1.0)
            if (counter == 0) or (torch.isnan(rmse_avg) > 0):
                print("Error calculating RMSE (val:%f , sum:%d)" % (rmse_avg, counter))
                plot_viz.append_loss(epoch + 1, epoch + 1, torch.tensor(0.0), "rmse", mode='test')
            else:
                rmse_avg /= counter
                print("Testing epoch {}: RMSE = {}".format(epoch+1, rmse_avg))            
                plot_viz.append_loss(epoch + 1, epoch + 1, rmse_avg, "rmse", mode='test')
        torch.enable_grad()
        model.train()        
            