#*
#    SLAM.py: the implementation of SLAM
#    created and maintained by Ty Nguyen
#    tynguyen@seas.upenn.edu
#    Feb 2020
#*
# from google.colab.patches import cv2_imshow
from scipy.special import logsumexp

import numpy as np
from numpy import cos,sin
import matplotlib.pyplot as plt
import load_data as ld
import os, sys, time
import p3_util as ut
from read_data import LIDAR, JOINTS
import probs_utils as prob
import math
import cv2
import transformations
from importlib import reload
reload(transformations)
import transformations as tf
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import logging
if (sys.version_info > (3, 0)):
    import pickle
else:
    import cPickle as pickle

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
interval = 1

class SLAM(object):
    def __init__(self):
        self._characterize_sensor_specs()
    
    def _read_data(self, src_dir, dataset=0, split_name='train'):
        self.dataset_= str(dataset)
        if split_name.lower() not in src_dir:
            src_dir  = src_dir + '/' + split_name
        print('\n------Reading Lidar and Joints (IMU)------')
        self.lidar_  = LIDAR(dataset=self.dataset_, data_folder=src_dir, name=split_name + '_lidar'+ self.dataset_)
        print ('\n------Reading Joints Data------')
        self.joints_ = JOINTS(dataset=self.dataset_, data_folder=src_dir, name=split_name + '_joint'+ self.dataset_)
        
        self.num_data_ = len(self.lidar_.data_)
        # Position of odometry
        self.odo_indices_ = np.empty((2,self.num_data_),dtype=np.int64)
        lidar_data = self.lidar_.data_
        # remove bias for odometry, init pose is (0,0,0)
        yaw_bias = lidar_data[0]['rpy'][0,2]
        pose_bias = lidar_data[0]['pose'][0,:2]
        for i in range(len(lidar_data)):
            lidar_data[i]['rpy'][0,2] -= yaw_bias
            lidar_data[i]['pose'][0,:2] -= pose_bias
        self.lidar_.data_ = lidar_data
    def _characterize_sensor_specs(self, p_thresh=None):
        # High of the lidar from the ground (meters)
        self.h_lidar_ = 0.93 + 0.33 + 0.15
        # Accuracy of the lidar
        self.p_true_ = 9
        self.p_false_ = 1.0/9
        
        #TODO: set a threshold value of probability to consider a map's cell occupied  
        self.p_thresh_ = 0.6 if p_thresh is None else p_thresh # > p_thresh => occupied and vice versa
        # Compute the corresponding threshold value of logodd
        self.logodd_thresh_ = prob.log_thresh_from_pdf_thresh(self.p_thresh_)
        

    def _init_particles(self, num_p=100, mov_cov=None, particles=None, weights=None, percent_eff_p_thresh=None):
        # Particles representation
        self.num_p_ = num_p
        #self.percent_eff_p_thresh_ = percent_eff_p_thresh
        self.particles_ = np.zeros((3,self.num_p_),dtype=np.float64) if particles is None else particles
        
        # Weights for particles
        self.weights_ = 1.0/self.num_p_*np.ones(self.num_p_) if weights is None else weights

        # Position of the best particle after update on the map
        self.best_p_indices_ = np.zeros((2,self.num_data_),dtype=np.int64)
        #self.best_p_indices_[:,0] = np.zeros(2)
        # Best particles
        self.best_p_ = np.zeros((3,self.num_data_))
        #self.best_p_[:,0] = np.zeros(3)
        # Corresponding time stamps of best particles
        self.time_ =  np.empty(self.num_data_)
       
        # Covariance matrix of the movement model
        tiny_mov_cov   = np.array([[1e-8, 0, 0],[0, 1e-8, 0],[0, 0 , 1e-8]])
        self.mov_cov_  = mov_cov if mov_cov is not None else tiny_mov_cov
        # To generate random noise: x, y, z = np.random.multivariate_normal(np.zeros(3), mov_cov, 1).T
        # this return [x], [y], [z]

        # Threshold for resampling the particles
        self.percent_eff_p_thresh_ = percent_eff_p_thresh

    def _init_map(self, map_resolution=0.05):
        '''*Input: resolution of the map - distance between two grid cells (meters)'''
        # Map representation
        MAP= {}
        MAP['res']   = map_resolution #meters
        MAP['xmin']  = -30  #meters
        MAP['ymin']  = -30
        MAP['xmax']  =  30
        MAP['ymax']  =  30
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #total cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1)) #total cells
        belief = 0.7
        MAP['occ_d'] = np.log(belief/(1-belief))
        MAP['free_d'] = np.log((1-belief)/belief)*.5
        occ_thres = 0.9
        free_thres = 0.2
        MAP['occ_thres'] = prob.log_thresh_from_pdf_thresh(occ_thres)
        MAP['free_thres'] = prob.log_thresh_from_pdf_thresh(free_thres)
        MAP['bound'] = 100 # allow log odds recovery
        MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=float) #DATA TYPE: char or int8
        # MAP['map'] = np.random.randint(-100,100,size=[MAP['sizex'],MAP['sizey']]).astype(float)
        self.MAP_ = MAP

        self.log_odds_ = np.zeros((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.float64)
        self.occu_ = np.ones((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.float64)
        # Number of measurements for each cell
        self.num_m_per_cell_ = np.zeros((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.uint64)


    def _build_first_map(self,t0=0,use_lidar_yaw=True):
        """Build the first map using first lidar and plot it"""
        self.t0 = t0
        # Extract a ray from lidar data, transform it to x-y-z frame
        print('\n--------Doing build the first map--------')
        lidar_idx = t0
        lidar_scan = self.lidar_.data_[lidar_idx]['scan']
        num_beams = lidar_scan.shape[1]
        lidar_angles = np.linspace(start=-135*np.pi/180, stop=135*np.pi/180, num=num_beams).reshape(1,-1)
        Pose = self.particles_[:, np.argmax(self.weights_)]
        selected_range = np.logical_and(lidar_scan>0.1, lidar_scan<30) # lidar spec
        lidar_scan_seleted_range = lidar_scan[selected_range]
        lidar_angles_selected_range = lidar_angles[selected_range]
        x_lidar = lidar_scan_seleted_range * cos(lidar_angles_selected_range)
        y_lidar = lidar_scan_seleted_range * sin(lidar_angles_selected_range)
        z_lidar = np.zeros(len(lidar_scan_seleted_range))
        lidar_selected_hit = np.vstack((x_lidar,y_lidar,z_lidar))# 3*n

        # find closest joint data(synchronization)
        joint_idx = np.argmin(np.abs(self.joints_.data_['ts']-self.lidar_.data_[lidar_idx]['t']))
        joint_angles = self.joints_.data_['head_angles'][:,joint_idx]

        # transform hit from lidar to world coordinate, also remove ground hitting
        world_hit = tf.lidar2world(lidar_selected_hit, joint_angles,self.lidar_.data_[lidar_idx]['rpy'][0,:],pose=Pose)
        occ = tf.world2map(world_hit[:2],self.MAP_)
        # update log odds for occupied grid, Note: pixels access should be (column, row)
        self.MAP_['map'][occ[1], occ[0]] += self.MAP_['occ_d']-self.MAP_['free_d'] # will add back later
        # update log odds for free grid, using contours to mask region between pose and hit
        mask = np.zeros(self.MAP_['map'].shape)
        contour = np.hstack((tf.world2map(Pose[:2],self.MAP_).reshape(-1,1), occ))
        cv2.drawContours(image=mask, contours = [contour.T], contourIdx = -1, color = self.MAP_['free_d'], thickness=-1)
        self.MAP_['map'] += mask
        # keep log odds within boundary, to allow recovery
        self.MAP_['map'][self.MAP_['map']>self.MAP_['bound']] = self.MAP_['bound']
        self.MAP_['map'][self.MAP_['map']<-self.MAP_['bound']] = -self.MAP_['bound']
        # print(self.MAP_['map'])

        # plot the first map
        h, w = self.MAP_['map'].shape
        Plot = np.zeros((h,w,3),np.uint8)
        # Trajectory = []
        occ_mask = self.MAP_['map']>self.MAP_['occ_thres']
        free_mask = self.MAP_['map']<self.MAP_['free_thres']
        und_mask = np.logical_not(np.logical_or(occ_mask, free_mask))
        Plot[occ_mask] = [0,0,0]            # black for occ
        Plot[free_mask] = [255,255,255]     # white for free
        Plot[und_mask] = [128,128,128]      # gray for und
        Plot[occ[1], occ[0]] = [0, 255, 0]  # green for lidar
        #cv2.imshow('SLAM',Plot)
        #cv2.waitKey(10)


    def _predict(self,t,use_lidar_yaw=False):
        logging.debug('\n-------- Doing prediction at t = {0}------'.format(t))
        # use yaw data from IMU rpy or use yaw data from odometry theta
        if use_lidar_yaw:
            curr_theta = self.lidar_.data_[t]['pose'][0,2]
            prev_theta = self.lidar_.data_[t-interval]['pose'][0,2]
        else:
            curr_theta = self.lidar_.data_[t]['rpy'][0,2]
            prev_theta = self.lidar_.data_[t-interval]['rpy'][0,2]
        d_theta = curr_theta - prev_theta
        # print('d_theta=',d_theta)
        world_2_body_rot = np.array([[np.cos(prev_theta), -np.sin(prev_theta)],
                            [np.sin(prev_theta), np.cos(prev_theta)]])
        # world_2_body_rot = tf.rot_z_axis(prev_theta)[:-1,:-1]   #2*2
        curr_xy = self.lidar_.data_[t]['pose'][0,:2]
        prev_xy = self.lidar_.data_[t-interval]['pose'][0,:2]
        d_xy_in_world = (curr_xy-prev_xy).reshape((-1,1))
        # print('d_xy in world=',d_xy_in_world)
        # relative movement in local frame, odom measurement is in global frame
        d_xy_in_body = np.dot(world_2_body_rot.T, d_xy_in_world)
        # print('d_xy=',d_xy_in_body)
        # apply relative movement and convert to global frame
        world_2_body_rots = np.array([[np.cos(self.particles_[2]), -np.sin(self.particles_[2])],
                                    [np.sin(self.particles_[2]), np.cos(self.particles_[2])]])
        # print('R_global=',world_2_body_rots)
        self.particles_[:2] += np.squeeze(np.einsum('ijk,il->ilk', world_2_body_rots, d_xy_in_body))
        self.particles_[2] += d_theta
        # print('t=',t,'particle=',self.particles_)
        # apply noise, set or use tiny_cov
        self.mov_cov_ = np.array([[0.001,0,0],[0,0.001,0],[0,0,0.001]])
        noise = np.random.multivariate_normal(np.zeros(3), self.mov_cov_, size=self.num_p_).T
        # # self.particles_[:2] += np.squeeze(np.einsum('ijk,ik->jk', world_2_body_rots, noise[:2]))
        # # self.particles_[2] += noise[2]
        self.particles_ += noise        # slightly incorrect but faster??
        # print('add noise')
        # print(f't={t},self.particles_={self.particles_}')

    def _update(self,t,t0=0,fig='on'):
        if t == t0:
            self._build_first_map(t0,use_lidar_yaw=True)
            return
        else:
            #######################################################################################
            # UPDATE MAP
            ######################################################################################
            lidar_scan = self.lidar_.data_[t]['scan']
            num_beams = lidar_scan.shape[1]
            lidar_angles = np.linspace(start=-135*np.pi/180, stop=135*np.pi/180, num=num_beams).reshape(1,-1)
            selected_range = np.logical_and(lidar_scan>0.1, lidar_scan<30) # lidar spec
            # print('selected range=',np.where(selected_range==False))
            lidar_scan_seleted_range = lidar_scan[selected_range]
            lidar_angles_selected_range = lidar_angles[selected_range]
            x_lidar = lidar_scan_seleted_range * cos(lidar_angles_selected_range)
            y_lidar = lidar_scan_seleted_range * sin(lidar_angles_selected_range)
            z_lidar = np.zeros(len(lidar_scan_seleted_range))
            lidar_selected_hit = np.vstack((x_lidar,y_lidar,z_lidar))# 3*n
            # find closest joint data(synchronization)
            joint_idx = np.argmin(np.abs(self.joints_.data_['ts']-self.lidar_.data_[t]['t']))
            joint_angles = self.joints_.data_['head_angles'][:,joint_idx]
            # transform hit from lidar to world coordinate, also remove ground hitting
            self.best_p_[:,t] = self.particles_[:,np.argmax(self.weights_)]
            # print(f't={t},pose={self.best_p_[:,t]}')
            world_hit = tf.lidar2world(lidar_selected_hit, joint_angles,self.lidar_.data_[t]['rpy'][0,:], pose=self.best_p_[:,t])
            # print('t=',t,'world_hit=',world_hit)
            occ = tf.world2map(world_hit[:2],self.MAP_)
            # update log odds for occupied grid, Note: pixels access should be (column, row)
            self.MAP_['map'][occ[1], occ[0]] += self.MAP_['occ_d']-self.MAP_['free_d'] # will add back later
            # update log odds for free grid, using contours to mask region between pose and hit
            mask = np.zeros(self.MAP_['map'].shape)
            best_particle_map = tf.world2map(self.best_p_[:2,t],self.MAP_).reshape(-1,1)
            contour = np.hstack((best_particle_map, occ))
            cv2.drawContours(image=mask, contours = [contour.T], contourIdx = -1, color = self.MAP_['free_d'], thickness=-1)
            self.MAP_['map'] += mask
            # keep log odds within boundary, to allow recovery
            self.MAP_['map'][self.MAP_['map']>self.MAP_['bound']] = self.MAP_['bound']
            self.MAP_['map'][self.MAP_['map']<-self.MAP_['bound']] = -self.MAP_['bound']
            # print('map=',np.where(self.MAP_['map']>0))
            ############################################################################################
            #UPDATE PARTICLES
            ############################################################################################
            # convert each particle into world frame
            particles_hit = tf.lidar2world(lidar_selected_hit,joint_angles,self.lidar_.data_[t]['rpy'][0,:],Particles=self.particles_)
            # get matching between map and particle lidar reading
            corr = np.zeros(self.num_p_)
            for i in range(self.num_p_):
                occ = tf.world2map(particles_hit[:2,:,i], self.MAP_)
                corr[i] = np.sum(self.MAP_['map'][occ[1],occ[0]]>self.MAP_['occ_thres'])
            corr /= 10 # by divide, adding a temperature to the softmax function
            # update particle weights
            log_weights = np.log(self.weights_) + corr
            log_weights -= np.max(log_weights) + logsumexp(log_weights - np.max(log_weights))
            # print(f'log_weights={log_weights}')
            self.weights_ = np.exp(log_weights)
            self.best_p_[:,t] = self.particles_[:,np.argmax(self.weights_)]
            occ = tf.world2map(self.best_p_[:2,t],self.MAP_)
            self.best_p_indices_[:,t] = np.array([occ[1,0],occ[0,0]])

            # print(f'best_p_indices={self.best_p_indices_[:,t]}')
        MAP = self.MAP_
        return MAP
    
