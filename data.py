
# coding: utf-8

# In[ ]:


import numpy as np
import sys
import os
import math
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions


# In[ ]:




class Data():
    
    def __init__(self):
        
        self.BIN_PATH = None
        self.CALIB_PATH = None
        self.LABEL_PATH = None
        self.CAM2_PATH = None
        
        
    def filter_camera_angle(self, places):
        """
            Filter camera angles for KiTTI Datasets
        """
        bool_in = np.logical_and((places[:, 1] < places[:, 0] - 0.27), (-places[:, 1] < places[:, 0] - 0.27))
        # bool_in = np.logical_and((places[:, 1] < places[:, 0]), (-places[:, 1] < places[:, 0]))
        return places[bool_in]
        
    
    def load_pc_from_bin(self, bin_path):
        """
            Loads point cloud data stored in bin format.
        """
        num_select = 2048
        selected_points =[]
        obj = np.fromfile(bin_path, dtype = np.float32).reshape(-1,4)
        pc = self.filter_camera_angle(obj)
        index = np.random.choice(pc.shape[0], num_select, replace=False)
        for i in range(len(index)):
            selected_points.append(pc[index[i]][0:3])
        selected_points = np.array(selected_points).reshape(1,-1,3)# return N*3 array
        return selected_points
    
    
    def get_boxcorners(self, places, rotates, size):
        """
            Create 8 corners of bounding box from bottom center.
        """
        corners = []
        for place, rotate, sz in zip(places, rotates, size):
            x, y, z = place
            h, w, l = sz
            if l > 10:
                continue

            corner = np.array([
                [x - l / 2., y - w / 2., z],
                [x + l / 2., y - w / 2., z],
                [x - l / 2., y + w / 2., z],
                [x - l / 2., y - w / 2., z + h],
                [x - l / 2., y + w / 2., z + h],
                [x + l / 2., y + w / 2., z],
                [x + l / 2., y - w / 2., z + h],
                [x + l / 2., y + w / 2., z + h],
            ])

            corner -= np.array([x, y, z])

            rotate_matrix = np.array([
                [np.cos(rotate), -np.sin(rotate), 0],
                [np.sin(rotate), np.cos(rotate), 0],
                [0, 0, 1]
            ])

            a = np.dot(corner, rotate_matrix.transpose())
            a += np.array([x, y, z])
            corners.append(a)
        return np.array(corners)
    
    
    def proj_to_velo(self, calib_data):
        """
            Projection matrix to 3D axis for 3D Label
        """
        rect = calib_data["R0_rect"].reshape(3, 3)
        velo_to_cam = calib_data["Tr_velo_to_cam"].reshape(3, 4) 
        inv_rect = np.linalg.inv(rect)
        inv_velo_to_cam = np.linalg.pinv(velo_to_cam[:, :3])
        return np.dot(inv_velo_to_cam, inv_rect)
    
    
    
    def read_calib_file(self, calib_path):
        """
           Reads a calibration file and splits it into data for the process. 
           The calibration files are txt files consisting of 7 matrices : P_0, 
           P_1, P_3 , P_4, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo. Each of 
           these matrices are extracted and stored in 'data' for a single .txt file.
        """
        
        data = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if not line or line == "\n":
                    continue
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data
    
    
    def read_label_from_txt(self, label_path):
        """
            Reads label from txt file. The label files are .txt files with 
            the following information:
            (1) Class - (Car, Pedestrian, Cyclist, etc.) at index 0
            (2) height of bounding box at index 8
            (3) width of bounding box at index 9
            (4) length of bounding box at index 10
            (5) x value of center of bounding box at index 11
            (6) y value of center of bounding box at index 12
            (7) z value of center of bounding box at index 13
            (8) heading (theta_y) of the object at index 14
        """
        text = np.fromfile(self, label_path)
        bounding_box = []
        with open(label_path, "r") as f:
            labels = f.read().split("\n")
            for label in labels:
                if not label:
                    continue
                label = label.split(" ")
                if (label[0] == "DontCare"):
                    continue
                y_class = [int(label[0]=="Car"), int(label[0]=="Van"), int(label[0]=="Pedestrian")]
                if label[0] == ("Car") or label[0]=="Van" or label[0]=="Pedestrian": #  or "Truck"
                    y_labels = list(np.hstack((label[8:15], y_class)))
                    bounding_box.append(y_labels)
                    break

        if bounding_box:
            data = np.array(bounding_box, dtype=np.float32)
            return data[:, 3:6], data[:, :3], data[:, 6], data[:,7:] 
        else:
            return None, None, None, None
    
    def read_labels(self, label_path, label_type, calib_path=None, is_velo_cam=False, proj_velo=None):
        """
            Read labels from txt file.
            Original Label value is shifted about 0.27m from object center.
            So need to revise the position of objects.
        """
        if label_type == "txt": #TODO
            places, size, rotates, y_class = read_label_from_txt(label_path)
            if places is None:
                return None, None, None, None
            rotates = np.pi / 2 - rotates
            dummy = np.zeros_like(places)
            dummy = places.copy()
            if calib_path:
                places = np.dot(dummy, proj_velo.transpose())[:, :3]
            else:
                places = dummy
            if is_velo_cam:
                places[:, 0] += 0.27

        data_combined = []
        for p, r, s , cl in zip(places, rotates, size, y_class):
            ps = np.hstack((cl[:],p[:],s[:]))
            data_combined.append(list(np.append(ps, r)))

        return places, rotates, size, y_class
    
    
    def center_to_sphere(self, places, size, resolution=0.50, min_value=np.array([0., -50., -4.5]), scale=4, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5)):
        """
            Convert object label to Training label for objectness loss.
        """
        x_logical = np.logical_and((places[:, 0] < x[1]), (places[:, 0] >= x[0]))
        y_logical = np.logical_and((places[:, 1] < y[1]), (places[:, 1] >= y[0]))
        z_logical = np.logical_and((places[:, 2] < z[1]), (places[:, 2] >= z[0]))
        xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))
        center = places.copy()
        center[:, 2] = center[:, 2] + size[:, 0] / 2.
        sphere_center = ((center[xyz_logical] - min_value) / (resolution * scale)).astype(np.int32)
        return sphere_center
    
    def sphere_to_center(self, p_sphere, resolution=0.5, scale=4, min_value=np.array([0., -50., -4.5])):
        """
            from sphere center to label center.
        """
        center = p_sphere * (resolution*scale) + min_value
        return center
    
    def pointcloud(self, velodyne_path, label_path=None, calib_path=None, dataformat="bin", label_type="txt", is_velo_cam=False):
        """
            Generates training data from raw dat.
        """
        
        p = []
        pc = None
        bounding_boxes = None
        places = None
        rotates = None
        size = None
        proj_velo = None

        filenames_velo = [d for d in sorted(os.listdir(velodyne_path)) if d[0]!='.']
        train_points = None
        train_labels = None
        train_classes = None
        for d in filenames_velo:
            value = d[0:6]
            print(value)
            velo_path = velodyne_path + value + '.bin'
            cal_path = calib_path + value + '.txt'
            lab_path = label_path + value + '.txt'
            
            cur_points = load_pc_from_bin(velo_path)
            cur_calib = read_calib_file(cal_path)
            
            proj_velo = proj_to_velo(cur_calib)[:, :3]
            places, rotates, size, classes = read_labels(lab_path, label_type, cal_path, 
                                                         is_velo_cam=is_velo_cam, proj_velo=proj_velo)
            corners = get_boxcorners(places, rotates, size)
            if places is None:
                continue
            if train_points is None:
                train_labels = corners
                train_points = cur_points
                train_classes = classes
            else:
                train_points = np.concatenate((train_points, cur_points), axis = 0)
                train_labels = np.concatenate((train_labels, corners), axis = 0)
                train_classes = np.concatenate((train_classes, classes), axis =0)

        return train_points, train_labels, train_classes

    
    def image(img_path):
        filenames_imgs = [img_path+d for d in sorted(os.listdir(img_path)) if d[0]!='.']
        batch_images = None
        for d in filenames_imgs:
            print(d)
            original = load_img(d, target_size=(224, 224))
            numpy_image = img_to_array(original)
            image_batch_cur = np.expand_dims(numpy_image, axis=0)
            if batch_images is None:
                batch_images = image_batch_cur
            else:
                batch_images = np.concatenate((batch_images,image_batch_cur ), axis = 0)
                
        return batch_images

