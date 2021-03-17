import visdom
import numpy
import torch
import datetime
from json2html import *

class NullVisualizer(object):
    def __init__(self):
        self.name = __name__

    def append_loss(self, epoch, global_iteration, loss, mode='train'):
        pass

    def show_images(self, images, title):
        pass
    
    def update_epoch(self, epoch):
        pass 

class VisdomPlotVisualizer(object):
    def __init__(self, name, server="http://localhost"):
        self.visualizer = visdom.Visdom(server=server, port=8097, env=name,\
            use_incoming_socket=False)
        self.name = name
        self.server = server
        self.first_train_value = True
        self.first_test_value = True
        self.plots = {}                                
        
    def append_loss(self, epoch, global_iteration, loss, loss_name="total", mode='train'):
        plot_name = loss_name + ('_loss' if mode == 'train' else '_error')
        opts = (
            {
                'title': plot_name,
                'xlabel': 'iterations',
                'ylabel': loss_name
            })
        loss_value = float(loss.detach().cpu().numpy())
        if loss_name not in self.plots:
            self.plots[loss_name] = self.visualizer.line(X=numpy.array([global_iteration]),\
                Y=numpy.array([loss_value]), opts=opts)
        else:
            self.visualizer.line(X=numpy.array([global_iteration]),\
                Y=numpy.array([loss_value]), win=self.plots[loss_name], name=mode, update='append')

    def config(self, **kwargs):
        self.visualizer.text(json2html.convert(json=dict(kwargs)))
    
    def update_epoch(self, epoch):
        pass 

class VisdomImageVisualizer(object):
    def __init__(self, name, server="http://localhost", count=2):
        self.name = name
        self.server = server
        self.count = count

    def update_epoch(self, epoch):
        self.visualizer = visdom.Visdom(server=self.server, port=8097,\
            env=self.name + str(epoch), use_incoming_socket=False)    

    def show_separate_images(self, images, title):
        b, c, h, w = images.size()
        take = self.count if self.count < b else b
        recon_images = images.detach().cpu()[:take, [2, 1, 0], :, :]\
            if c == 3 else images.detach().cpu()[:take, :, :, :]
        for i in range(take):
            img = recon_images[i, :, :, :]
            opts = (
            {
                'title': title + "_" + str(i),
                 'width': w, 'height': h
            })
            self.visualizer.image(img, opts=opts,\
                win=self.name + title + "_window_" + str(i))

    def show_map(self, maps, title):
        b, c, h, w = maps.size()        
        maps_cpu = torch.flip(maps, dims=[2]).detach().cpu()[:self.count, :, :, :]
        for i in range(min(b, self.count)):
            opts = (
            {
                'title': title + str(i), 'colormap': 'Viridis'
            })
            heatmap = maps_cpu[i, :, :, :].squeeze(0)
            #TODO: flip images before heatmap call
            self.visualizer.heatmap(heatmap,\
                opts=opts, win=self.name + title + "_window_" + str(i))      
