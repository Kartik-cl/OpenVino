# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:46:55 2020

@author: RujutaPimprikar
"""


import cv2
import os
import numpy as np
import datetime
import requests
import socket
from common import (CalibrationParams, convert_depth_uint_to_float)
import base64
from PIL import Image
import io
import json
from MaskRCNNInfer import Inference
import subprocess
import importlib.util
import cherrypy

def triangle_area(x1,y1,z1,x2,y2,z2,x3,y3,z3):
    '''
    Function to obtain area of 3D triangle
    '''
    v12 = [x2-x1, y2-y1, z2-z1]
    v13 = [x3-x1, y3-y1, z3-z1]
    cross_prod = np.cross(v12, v13)
    magnitude = np.sqrt(cross_prod.dot(cross_prod))
    return 0.5*magnitude

def poly_tri_area(poly):
    '''
    Function to calculate area of polygon, by connecting one vertex to all other vertices to obtain triangles
    '''
    if len(poly) < 3: # not a plane
        return 0
    total = 0
    N = len(poly)
    v0 = poly[0]
    for i in range(2,N):
        v1 = poly[i-1]
        v2 = poly[i]
        total += triangle_area(v0[0],v0[1],v0[2],v1[0],v1[1],v1[2],v2[0],v2[1],v2[2])
    return total


def unit_normal(a, b, c):
    '''
    Calculate unit normal vector of plane defined by points a, b, and c
    '''
    x = np.linalg.det([[1,a[1],a[2]],
         [1,b[1],b[2]],
         [1,c[1],c[2]]])
    y = np.linalg.det([[a[0],1,a[2]],
         [b[0],1,b[2]],
         [c[0],1,c[2]]])
    z = np.linalg.det([[a[0],a[1],1],
         [b[0],b[1],1],
         [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)



def poly_area(poly):
    '''
    Function to calculate area of polygon assuming it is 3D coplanar
    '''
    if len(poly) < 3: # not a plane
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i+1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)


def is_occluded_region(x1, y1, z1):
    if abs(x1) == 0 and abs(y1) == 0 and abs(z1) == 0:
        return True
    return False


def get_euclidean_distance(x1, x2, y1, y2, z1=0, z2=0):
    return np.sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1) + (z2-z1)*(z2-z1))


def image_registration(float_depth_image, rgb_shape, calib_params, foo):
    depth_points_3d = foo.rgbd.depthTo3d(float_depth_image, calib_params.depth_intrinsics)
    depth_points_in_rgb_frame = cv2.perspectiveTransform(depth_points_3d, calib_params.depth_extrinsics)

    fx = calib_params.rgb_intrinsics[0, 0]
    fy = calib_params.rgb_intrinsics[1, 1]
    cx = calib_params.rgb_intrinsics[0, 2]
    cy = calib_params.rgb_intrinsics[1, 2]

    float_depth_registered = np.zeros(rgb_shape, dtype='float')

    for points in depth_points_in_rgb_frame:
        for point in points:
            u = int(fx * point[0] / point[2] + cx)
            v = int(fy * point[1] / point[2] + cy)

            height = rgb_shape[0]
            width = rgb_shape[1]
            if (u >= 0 and u < width and v >= 0 and v < height):
                float_depth_registered[v, u] = point[2]

    kernel = np.ones((3, 3), np.uint16)
    float_depth_registered = cv2.morphologyEx(float_depth_registered,
                                              cv2.MORPH_CLOSE, kernel)
    
    return float_depth_registered


def get_dimensions(calib_params, depth_image, image, available_points_threshold=0.5):
    
    print(calib_params.z_scaling)
    print(calib_params.depth_scale)
    print(depth_image.shape)
    float_depth_image = convert_depth_uint_to_float(depth_image, calib_params.z_scaling, calib_params.depth_scale)
    
    spec = importlib.util.spec_from_file_location("cv2", "/usr/local/lib/python3.6/dist-packages/cv2/cv2.cpython-36m-x86_64-linux-gnu.so")
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    float_registered_depth_image = image_registration(float_depth_image, (calib_params.rgb_height, calib_params.rgb_width), calib_params, foo)
    depth_points_3d = foo.rgbd.depthTo3d(float_registered_depth_image, calib_params.rgb_intrinsics)
    obj = Inference()
    start_time = datetime.datetime.now()
    response = obj.openVino(image)
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time).total_seconds()
    print(f'Time taken : {time_diff} seconds')
    if len(response['predictions'])>0:
        for i in range(len(response['predictions'])): #check sanity for label and vertices indexes
            poly_points = []
            num_points = len(response['vertices'][i])
            label = response['predictions'][i]
            print('-------------')
            print(label)
            for poly in response['vertices'][i]:
                
                i, j = poly[1], poly[0]  # y,x
                point_x1 = depth_points_3d[i, j, 0] 
                point_y1 = depth_points_3d[i, j, 1]
                point_z1 = depth_points_3d[i, j, 2]
                  
                if is_occluded_region(point_x1, point_y1, point_z1):
                    print(poly[0], poly[1])
                    print('3d point in occluded region')
                    pass
                else: 
                    poly_points.append([point_x1, point_y1, point_z1])
            
            available_points = len(poly_points)/num_points
            print(len(poly_points), num_points, available_points)
            
            if available_points >= available_points_threshold:
                polygon_area = poly_tri_area(poly_points) # poly_area(poly_points)
                print('Area of polygon (m^2)', polygon_area) 
                print('Area of polygon (cm^2)', polygon_area*100*100) 
            else:
                print('Too many points in occluded region hence cannot estimate area')
            
                    
                   
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time).total_seconds()
    print(f'Time taken : {time_diff} seconds')
    


config = {
    'global' : {
        'server.socket_host' : '127.0.0.1',
        'server.socket_port' : 3066,
        'server.max_request_body_size' : 0,
        'server.max_request_header_size' : 0
    }
}


class App:
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def index(self, depth_image, image):
        use_stereo_depth = False
        calib_params = CalibrationParams()
        d435_calib_file = 'realsense_d435_device_depth.yaml'
        if use_stereo_depth:
            d435_depth_x_offset_pixels = 0
            d435_depth_y_offset_pixels = 0
        else:
            d435_depth_x_offset_pixels = -3
            d435_depth_y_offset_pixels = -1
    
        calib_params.read_from_yaml(d435_calib_file)
        calib_params.depth_intrinsics[0, 2] += d435_depth_x_offset_pixels
        calib_params.depth_intrinsics[1, 2] += d435_depth_y_offset_pixels
        di_data = base64.b64decode(depth_image)
        depth_image = Image.open(io.BytesIO(di_data))
        depth_image = np.array(depth_image)
        data = base64.b64decode(image)
        image = Image.open(io.BytesIO(data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        get_dimensions(calib_params, depth_image, image)
        input("hi")
        return "check server logs"




if __name__ == '__main__':
    cherrypy.quickstart(App(), '/', config) 




