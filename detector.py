import cv2
import numpy as np
import pandas as pd

class FiguresDetector:
    
    _figures = {
                0: "Triangle",
                1: "Circle",
                2: "Square"
              }
    def __init__(self):
        pass
    
    @staticmethod
    def normalize(img):
        img[img>127] = 255
        img[img<127] = 0
        return img
    
    def detect(self, img, show_stats=True):
        img = self.normalize(img)
        n_connected, areas, bboxes, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(img,
                                                                                      connectivity=4,
                                                                                      ltype=cv2.CV_16U,
                                                                                      ccltype=cv2.CCL_WU)
        n_figures = {
            "Triangle": 0,
            "Circle": 0,
            "Square": 0,
            "Total": n_connected-1
        }
        triangles = np.zeros_like(img, dtype=np.uint8)
        circles = np.zeros_like(img, dtype=np.uint8)
        squares = np.zeros_like(img, dtype=np.uint8)
        for i in range(1,n_connected):
            item = (areas==i).astype(np.uint8)
            h = np.linalg.norm(np.argwhere(item>0)-np.array([centroids[i][1], 
                                                             centroids[i][0]]),axis=1).max()
            area = np.sum(item)
            a_tcs = np.array([np.square(3/2*h)/np.sqrt(3),
                              np.pi*h**2, 
                              2*np.square(h)])
            
            fig_idx = np.argmin(np.abs(a_tcs-area))
            
            if self._figures[fig_idx] == 'Triangle':
                triangles = triangles + item
                n_figures['Triangle'] += 1
            elif self._figures[fig_idx] == 'Circle':
                circles = circles + item
                n_figures['Circle'] += 1
            elif self._figures[fig_idx] == 'Square':
                squares = squares + item
                n_figures['Square'] += 1
        assert n_figures['Triangle'] + n_figures['Square'] + n_figures['Circle'] == n_figures['Total'],\
            'Not all figures were detected successfully!'
        if show_stats:
            df = pd.DataFrame(data=n_figures, index=['Count'])
            print(df)
        
        out_result = np.stack([triangles, circles, squares], axis=2)*255
        return out_result
        