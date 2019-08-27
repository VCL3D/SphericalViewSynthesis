###################################
# 360 dataset pytorch dataloader
###################################
import os

import numpy as np
import cv2
import PIL.Image as Image
import datetime

import torch
from torch.utils.data import Dataset
from torchvision import transforms
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

############################################################################################################
# We use a text file to hold our dataset's filenames
# The "filenames list" has the following format 
#
# path/to/Left/rgb.png path/to/Right/rgb.png path/to/Up/rgb.png path/to/Left/depth.exr path/to/Right/depth.exr path/to/Up/depth.exr 
#
# We also have a Trinocular version, but you get the feeling.
#############################################################################################################

class Dataset360D(Dataset):
    #360D Dataset#
    def __init__(self, filenamesFile, delimiter, mode, inputShape, transform=None, rescaled=False):
        #########################################################################################################
        # Arguments:
        # -filenamesFile: Absolute path to the aforementioned filenames .txt file
        # -transform    : (Optional) transform to be applied on a sample
        # -mode         : Dataset mode. Available options: mono, lr (Left-Right), ud (Up-Down), tc (Trinocular)
        #########################################################################################################
        self.height = inputShape[0]
        self.width = inputShape[1]
        self.sample = {}                                # one dataset sample (dictionary)
        self.resize2 = transforms.Resize([128, 256])    # function to resize input image by a factor of 2
        self.resize4 = transforms.Resize([64, 128])     # function to resize input image by a factor of 4
        self.pilToTensor =  transforms.ToTensor() if transform is None else transforms.Compose((
        [
            transforms.ToTensor(),                      # function to convert pillow image to tensor
            transform        
        ])
        )
        self.filenamesFilePath = filenamesFile          # file containing image paths to load
        self.delimiter = delimiter                      # delimiter in filenames file        
        self.mode = mode                                # dataset mode
        self.initDict(self.mode)                        # initializes dictionary with filepaths
        self.loadFilenamesFile()                        # loads filepaths to dictionary
        self.rescaled = rescaled
        
    # Check if given dataset mode is correct
    # Available modes: mono, lr, ud, tc
    def checkMode(self, mode):
        accepted = False
        if (mode != "mono" and mode != "lr" and mode != "ud" and mode != "tc"):
            print("{} | Given dataset mode [{}] is not known. Available modes: mono, lr, ud, tc".format(datetime.datetime.now(), mode))
            exit()
        else:
            accepted = True
        return accepted

    # initializes dictionary's lists w.r.t. the dataset's mode
    def initDict(self, mode):
        if (mode == "mono"):
            self.sample["leftRGB"] = []
            self.sample["leftRGB2"] = []
            self.sample["leftRGB4"] = []
            self.sample["leftDepth"] = []
            self.sample["leftDepth2"] = []
            self.sample["leftDepth4"] = []
        elif (mode == "lr"):
            self.sample["leftRGB"] = []
            self.sample["leftRGB2"] = []
            self.sample["leftRGB4"] = []
            self.sample["rightRGB"] = []
            self.sample["rightRGB2"] = []
            self.sample["rightRGB4"] = []
            self.sample["leftDepth"] = []
            self.sample["leftDepth2"] = []
            self.sample["leftDepth4"] = []
            self.sample["rightDepth"] = []
            self.sample["rightDepth2"] = []
            self.sample["rightDepth4"] = []
        elif (mode == "ud"):
            self.sample["leftRGB"] = []
            self.sample["leftRGB2"] = []
            self.sample["leftRGB4"] = []
            self.sample["upRGB"] = []
            self.sample["upRGB2"] = []
            self.sample["upRGB4"] = []
            self.sample["leftDepth"] = []
            self.sample["leftDepth2"] = []
            self.sample["leftDepth4"] = []
            self.sample["upDepth"] = []
            self.sample["upDepth2"] = []
            self.sample["upDepth4"] = []
        elif (mode == "tc"):
            self.sample["leftRGB"] = []
            self.sample["leftRGB2"] = []
            self.sample["leftRGB4"] = []
            self.sample["rightRGB"] = []
            self.sample["rightRGB2"] = []
            self.sample["rightRGB4"] = []
            self.sample["upRGB"] = []
            self.sample["upRGB2"] = []
            self.sample["upRGB4"] = []
            self.sample["leftDepth"] = []
            self.sample["leftDepth2"] = []
            self.sample["leftDepth4"] = []
            self.sample["rightDepth"] = []
            self.sample["rightDepth2"] = []
            self.sample["rightDepth4"] = []
            self.sample["upDepth"] = []
            self.sample["upDepth2"] = []
            self.sample["upDepth4"] = []

    # configures samples when in mono mode
    # loads filepaths to dictionary's list
    def initModeMono(self, lines):
        for line in lines:
            leftRGBPath = line.split(self.delimiter)[0]
            leftDepthPath = line.split(self.delimiter)[3]
            self.sample["leftRGB"].append(leftRGBPath)
            self.sample["leftDepth"].append(leftDepthPath)


    # configures dataset samples when in Left-Right mode
    def initModeLR(self, lines):
        for line in lines:
            leftRGBPath = line.split(self.delimiter)[0]
            rightRGBPath = line.split(self.delimiter)[1]
            leftDepthPath = line.split(self.delimiter)[3]
            rightDepthPath = line.split(self.delimiter)[4]
            self.sample["leftRGB"].append(leftRGBPath)
            self.sample["rightRGB"].append(rightRGBPath)
            self.sample["leftDepth"].append(leftDepthPath)
            self.sample["rightDepth"].append(rightDepthPath)

    # configures dataset samples when in Up-Down mode
    def initModeUD(self, lines):
        for line in lines:
            leftRGBPath = line.split(self.delimiter)[0]
            upRGBPath = line.split(self.delimiter)[2]
            leftDepthPath = line.split(self.delimiter)[3]
            upDepthPath = line.split(self.delimiter)[5]
            self.sample["leftRGB"].append(leftRGBPath)
            self.sample["upRGB"].append(upRGBPath)
            self.sample["leftDepth"].append(leftDepthPath)
            self.sample["upDepth"].append(upDepthPath)

    # configures dataset samples when in Trinocular mode
    def initModeTC(self, lines):
        for line in lines:
            leftRGBPath = line.split(self.delimiter)[0]
            rightRGBPath = line.split(self.delimiter)[1]
            upRGBPath = line.split(self.delimiter)[2]
            leftDepthPath = line.split(self.delimiter)[3]
            rightDepthPath = line.split(self.delimiter)[4]
            upDepthPath = line.split(self.delimiter)[5]
            self.sample["leftRGB"].append(leftRGBPath)
            self.sample["rightRGB"].append(rightRGBPath)
            self.sample["upRGB"].append(upRGBPath)
            self.sample["leftDepth"].append(leftDepthPath)
            self.sample["rightDepth"].append(rightDepthPath)
            self.sample["upDepth"].append(upDepthPath)

    # Loads filenames from .txt file and saves the samples' paths w.r.t. the dataset mode
    def loadFilenamesFile(self):
        if (not os.path.exists(self.filenamesFilePath)):
            print("{} | Filepath [{}] does not exist.".format(datetime.datetime.now(), self.filenamesFilePath))
            exit()
        fileID = open(self.filenamesFilePath, "r")
        lines = fileID.readlines()
        if (lines == 0):
            print("{} | Cannot open file: {}".format(datetime.datetime.now(), self.filenamesFilePath))
            exit()
        self.length = len(lines)
        if (self.mode == "mono"):
            self.initModeMono(lines)
        elif (self.mode == "lr"):
            self.initModeLR(lines)
        elif (self.mode == "ud"):
            self.initModeUD(lines)
        elif (self.mode == "tc"):
            self.initModeTC(lines)

    # loads sample from dataset mono mode
    def loadItemMono(self, idx):
        item = {}
        if (idx >= self.length):
            print("Index [{}] out of range. Dataset length: {}".format(idx, self.length))
        else:
            dtmp = np.array(cv2.imread(self.sample["leftDepth"][idx], cv2.IMREAD_ANYDEPTH))            
            left_depth = torch.from_numpy(dtmp)
            left_depth.unsqueeze_(0)
            if self.rescaled:
                dtmp2 = cv2.resize(dtmp, (dtmp.shape[1] // 2, dtmp.shape[0] // 2))
                dtmp4 = cv2.resize(dtmp, (dtmp.shape[1] // 4, dtmp.shape[0] // 4))                
                left_depth2 = torch.from_numpy(dtmp2)
                left_depth2.unsqueeze_(0)
                left_depth4 = torch.from_numpy(dtmp4)
                left_depth4.unsqueeze_(0)

            pilRGB = Image.open(self.sample["leftRGB"][idx])
            rgb = self.pilToTensor(pilRGB)
            if self.rescaled:
                rgb2 = self.pilToTensor(self.resize2(pilRGB))
                rgb4 = self.pilToTensor(self.resize4(pilRGB))
            item = {
                "leftRGB": rgb,
                "leftRGB2": rgb2,
                "leftRGB4": rgb4, 
                "leftDepth": left_depth,
                "leftDepth2": left_depth2,
                "leftDepth4": left_depth4,                
                "leftDepth_filename": os.path.basename(self.sample["leftDepth"][idx][:-4])
                } if self.rescaled else {
                "leftRGB": rgb,
                "leftDepth": left_depth,
                "leftDepth_filename": os.path.basename(self.sample["leftDepth"][idx][:-4])
                }
        return item

    # loads sample from dataset lr mode
    def loadItemLR(self, idx):
        item = {}
        if (idx >= self.length):
            print("Index [{}] out of range. Dataset length: {}".format(idx, self.length))
        else:
            leftRGB = Image.open(self.sample["leftRGB"][idx])
            rightRGB = Image.open(self.sample["rightRGB"][idx])
            if self.rescaled:
                leftRGB2 = self.pilToTensor(self.resize2(leftRGB))
                leftRGB4 = self.pilToTensor(self.resize4(leftRGB))
                rightRGB2 = self.pilToTensor(self.resize2(rightRGB))
                rightRGB4 = self.pilToTensor(self.resize4(rightRGB))
            
            dtmp = np.array(cv2.imread(self.sample["leftDepth"][idx], cv2.IMREAD_ANYDEPTH))            
            left_depth = torch.from_numpy(dtmp)
            left_depth.unsqueeze_(0)
            if self.rescaled:
                dtmp2 = cv2.resize(dtmp, (dtmp.shape[1] // 2, dtmp.shape[0] // 2))
                dtmp4 = cv2.resize(dtmp, (dtmp.shape[1] // 4, dtmp.shape[0] // 4))            
                left_depth2 = torch.from_numpy(dtmp2)
                left_depth2.unsqueeze_(0)
                left_depth4 = torch.from_numpy(dtmp4)
                left_depth4.unsqueeze_(0)

            dtmp = np.array(cv2.imread(self.sample["rightDepth"][idx], cv2.IMREAD_ANYDEPTH))            
            right_depth = torch.from_numpy(dtmp)
            right_depth.unsqueeze_(0)
            if self.rescaled:
                tmp2 = cv2.resize(dtmp, (dtmp.shape[1] // 2, dtmp.shape[0] // 2))
                dtmp4 = cv2.resize(dtmp, (dtmp.shape[1] // 4, dtmp.shape[0] // 4))            
                right_depth2 = torch.from_numpy(dtmp2)
                right_depth2.unsqueeze_(0)
                right_depth4 = torch.from_numpy(dtmp4)
                right_depth4.unsqueeze_(0)
            item = { 
                "leftRGB": self.pilToTensor(leftRGB),
                "rightRGB": self.pilToTensor(rightRGB),
                "leftRGB2": leftRGB2,
                "rightRGB2": rightRGB2,
                "leftRGB4": leftRGB4,
                "rightRGB4": rightRGB4 ,
                "leftDepth": left_depth,
                'leftDepth2': left_depth2,
                'leftDepth4': left_depth4,
                "rightDepth": right_depth,
                "rightDepth2": right_depth2,
                "rightDepth4": right_depth4,
                'leftDepth_filename': os.path.basename(self.sample['leftDepth'][idx][:-4])
                } if self.rescaled else { 
                "leftRGB": self.pilToTensor(leftRGB),
                "rightRGB": self.pilToTensor(rightRGB),
                "leftDepth": left_depth,
                "rightDepth": right_depth,
                'leftDepth_filename': os.path.basename(self.sample['leftDepth'][idx][:-4])
                }
        return item
    
    # loads sample from dataset ud mode
    def loadItemUD(self, idx):
        item = {}
        if (idx >= self.length):
            print("Index [{}] out of range. Dataset length: {}".format(idx, self.length))
        else:
            leftRGB = Image.open(self.sample["leftRGB"][idx])
            upRGB = Image.open(self.sample["upRGB"][idx])
            if self.rescaled:
                leftRGB2 =  self.pilToTensor(self.resize2(leftRGB))
                leftRGB4 = self.pilToTensor(self.resize4(leftRGB))
                upRGB2 = self.pilToTensor(self.resize2(upRGB))
                upRGB4 = self.pilToTensor(self.resize4(upRGB))
            dtmp = np.array(cv2.imread(self.sample["leftDepth"][idx], cv2.IMREAD_ANYDEPTH))            
            depth = torch.from_numpy(dtmp)
            depth.unsqueeze_(0)
            if self.rescaled:
                dtmp2 = cv2.resize(dtmp, (self.width // 2, self.height // 2))
                dtmp4 = cv2.resize(dtmp, (self.width // 4, self.height // 4))            
                depth2 = torch.from_numpy(dtmp2)
                depth2.unsqueeze_(0)
                depth4 = torch.from_numpy(dtmp4)
                depth4.unsqueeze_(0)

            
            dtmp = np.array(cv2.imread(self.sample["upDepth"][idx], cv2.IMREAD_ANYDEPTH))            
            up_depth = torch.from_numpy(dtmp)
            up_depth.unsqueeze_(0)
            if self.rescaled:
                tmp2 = cv2.resize(dtmp, (dtmp.shape[1] // 2, dtmp.shape[0] // 2))
                dtmp4 = cv2.resize(dtmp, (dtmp.shape[1] // 4, dtmp.shape[0] // 4))            
                up_depth2 = torch.from_numpy(dtmp2)
                up_depth2.unsqueeze_(0)
                up_depth4 = torch.from_numpy(dtmp4)
                up_depth4.unsqueeze_(0)

            item = { 
                "leftRGB": self.pilToTensor(leftRGB),  
                "upRGB": self.pilToTensor(upRGB), 
                "leftRGB2": leftRGB2, 
                "upRGB2": upRGB2, 
                "leftRGB4": leftRGB4, 
                "upRGB4": upRGB4,
                "leftDepth": depth,
                "leftDepth2": depth2,
                "leftDepth4": depth4,
                "upDepth": up_depth,
                "upDepth2": up_depth2,
                "upDepth4": up_depth4,
                'leftDepth_filename': os.path.basename(self.sample['leftDepth'][idx][:-4])
            } if self.rescaled else { 
                "leftRGB": self.pilToTensor(leftRGB),  
                "upRGB": self.pilToTensor(upRGB), 
                "leftDepth": depth,
                "upDepth": up_depth,
                'leftDepth_filename': os.path.basename(self.sample['leftDepth'][idx][:-4])
            }
        return item

    # loads sample from dataset tc mode
    def loadItemTC(self, idx):
        item = {}
        if (idx >= self.length):
            print("Index [{}] out of range. Dataset length: {}".format(idx, self.length))
        else:
            leftRGB = Image.open(self.sample["leftRGB"][idx])
            rightRGB = Image.open(self.sample["rightRGB"][idx])
            upRGB = Image.open(self.sample["upRGB"][idx])
            if self.rescaled:
                leftRGB2 = self.pilToTensor(self.resize2(leftRGB))
                leftRGB4 = self.pilToTensor(self.resize4(leftRGB))
                rightRGB2 = self.pilToTensor(self.resize2(rightRGB))
                rightRGB4 = self.pilToTensor(self.resize4(rightRGB))
                upRGB2 = self.pilToTensor(self.resize2(upRGB))
                upRGB4 = self.pilToTensor(self.resize4(upRGB))

            dtmp = np.array(cv2.imread(self.sample["leftDepth"][idx], cv2.IMREAD_ANYDEPTH))            
            depth = torch.from_numpy(dtmp)
            depth.unsqueeze_(0)
            if self.rescaled:
                dtmp2 = cv2.resize(dtmp, (self.width // 2, self.height // 2))
                dtmp4 = cv2.resize(dtmp, (self.width // 4, self.height // 4))
                depth2 = torch.from_numpy(dtmp2)
                depth2.unsqueeze_(0)
                depth4 = torch.from_numpy(dtmp4)
                depth4.unsqueeze_(0)

            dtmp = np.array(cv2.imread(self.sample["rightDepth"][idx], cv2.IMREAD_ANYDEPTH))
            right_depth = torch.from_numpy(dtmp)
            right_depth.unsqueeze_(0)
            if self.rescaled:
                tmp2 = cv2.resize(dtmp, (dtmp.shape[1] // 2, dtmp.shape[0] // 2))
                dtmp4 = cv2.resize(dtmp, (dtmp.shape[1] // 4, dtmp.shape[0] // 4))            
                right_depth2 = torch.from_numpy(dtmp2)
                right_depth2.unsqueeze_(0)
                right_depth4 = torch.from_numpy(dtmp4)
                right_depth4.unsqueeze_(0)

            dtmp = np.array(cv2.imread(self.sample["upDepth"][idx], cv2.IMREAD_ANYDEPTH))
            up_depth = torch.from_numpy(dtmp)
            up_depth.unsqueeze_(0)
            if self.rescaled:
                tmp2 = cv2.resize(dtmp, (dtmp.shape[1] // 2, dtmp.shape[0] // 2))
                dtmp4 = cv2.resize(dtmp, (dtmp.shape[1] // 4, dtmp.shape[0] // 4))            
                up_depth2 = torch.from_numpy(dtmp2)
                up_depth2.unsqueeze_(0)
                up_depth4 = torch.from_numpy(dtmp4)
                up_depth4.unsqueeze_(0)

            item = { 
                "leftRGB": self.pilToTensor(leftRGB), 
                "rightRGB": self.pilToTensor(rightRGB), 
                "upRGB": self.pilToTensor(upRGB),
                "leftRGB2": leftRGB2,
                "rightRGB2": rightRGB2,
                "upRGB2": upRGB2,
                "leftRGB4": leftRGB4,
                "rightRGB4": rightRGB4,
                "upRGB4": upRGB4,
                "leftDepth": depth,
                "leftDepth2": depth2,
                "leftDepth4": depth4,
                "upDepth": up_depth,
                "upDepth2": up_depth2,
                "upDepth4": up_depth4,
                "rightDepth": right_depth,
                "rightDepth2": right_depth2,
                "rightDepth4": right_depth4,
                "depthFilename": os.path.basename(self.sample["leftDepth"][idx][:-4])
                } if self.rescaled else { 
                "leftRGB": self.pilToTensor(leftRGB), 
                "rightRGB": self.pilToTensor(rightRGB), 
                "upRGB": self.pilToTensor(upRGB),                                
                "leftDepth": depth,
                "rightDepth": right_depth,                
                "upDepth": up_depth,
                "depthFilename": os.path.basename(self.sample["leftDepth"][idx][:-4])
                }
        return item

    # torch override
    # returns samples length
    def __len__(self):
        return self.length

    # torch override
    def __getitem__(self, idx):
        if (self.mode == "mono"):
            return self.loadItemMono(idx)
        elif(self.mode == "lr"):
            return self.loadItemLR(idx)
        elif(self.mode == "ud"):
            return self.loadItemUD(idx)
        elif(self.mode == "tc"):
            return self.loadItemTC(idx)
        
        

          