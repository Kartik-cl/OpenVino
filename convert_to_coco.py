import os
import cv2
import xml.etree.ElementTree as ET
import pprint
import numpy as np
import shutil

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

pp = pprint.PrettyPrinter()
path = 'data/augmeted_defects_clubs_020/'

data = {}
data['info'] = {'url': 'http://cocodataset.org', 'contributor': 'COCO Consortium', 'version': '1.0', 'date_created': '2017/09/01', 'description': 'COCO 2017 Dataset', 'year': 2017}
data['images'] = []
data['annotations'] = []
data['categories'] = [{"id": 0, "name": "defect_on_side", "supercategory": "defect_on_side"}, {"id": 1, "name": "defect_on_top", "supercategory": "defect_on_top"}]
data['licenses'] = [{'name': 'dummy', 'url': 'dummy', 'id': 1}]


i = 0
j = 1
destination = 'data/coco/images/val2017'
for file in os.listdir(path):
	xml_file = file.replace("png", "xml")
	if(not os.path.exists(os.path.join(path, xml_file))):
		continue
	if(".png" in file):
		i+=1
		print(i)
		img = cv2.imread(os.path.join(path, file))
		shutil.copyfile(os.path.join(path, file), (os.path.join(destination, file)))
		rec = {}
		rec['id'] = i
		rec['license'] = 4
		rec['coco_url'] = "dummy"
		rec['flickr_url'] = "dummy"
		rec['width'] = img.shape[1]
		rec['height'] = img.shape[0]
		rec['file_name'] = file
		rec['date_captured'] = "dummy"
		data['images'].append(rec)	
		tree = ET.parse(os.path.join(path, xml_file))
		root = tree.getroot()
		for node in root.findall('object'):
			rec = {}
			bbox = []
			segmentation = []
			points = []
			for child in node:
				if child.tag == 'bndbox':
					bbox.append(int(child[0].text))
					bbox.append(int(child[1].text))
					bbox.append(int(child[2].text) - int(child[0].text))
					bbox.append(int(child[3].text) - int(child[1].text))
					rec['bbox'] = bbox
				if child.tag == 'segment_polygons':
					for kid in child:
						for kid1 in kid:
							points.append(int(kid1[0].text))
							points.append(int(kid1[1].text))
						segmentation.append(points)
					rec['segmentation'] = segmentation
					xs, ys, area = [], [], 0
					for seg in segmentation:
						for k in range(len(seg)):
							if(k%2 == 0):
								xs.append(seg[k])
							else:
								ys.append(seg[k])
						area+=PolyArea(np.array(xs),np.array(ys))
					rec['area'] = area
					rec['iscrowd'] = 0
				if child.tag == 'name':
					if child.text == 'defect_on_side':
						rec['category_id'] = 0
					else:
						rec['category_id'] = 1		
					rec['id'] = j	
				rec['image_id'] = i
			j+=1
			data['annotations'].append(rec)


import json
with open('data/coco/annotations/instances_val2017.json', 'w') as fp:
    json.dump(data, fp)	
		
