import cv2
import numpy as np
import sys, getopt
from enum import Enum

class dimension(Enum):
    N = 0
    H = 1
    W = 2
    C = 3

dim2dig = {
    "N": dimension.N, "n": dimension.N,
    "H": dimension.H, "h": dimension.H,
    "W": dimension.W, "w": dimension.W,
    "C": dimension.C, "c": dimension.C
                }

dig2dim = {
    dimension.N : "N",
    dimension.H : "H",
    dimension.W : "W",
    dimension.C : "C"
                }


   
class shape:
    def __init__(self, shape_str : str, layout_str : str):
        self.shape = np.array(shape_str.split("*")).astype(int)
        self._layout = self.layout_read(self.shape, layout_str)

    @staticmethod
    def layout_read(shape : np.array, layout_str: str):
        ret = dict()
        dims = len(shape)
        if dims != len(layout_str):
            raise Exception("dim unmatch with layout")
        for i in range(0,dims):
            ret[dim2dig[layout_str[i]]] = shape[i]
        return ret   

    def __str__(self):
        shape_str = ""
        for dim, shape in self._layout.items():
            shape_str += (dig2dim[dim] + ":" + str(shape) + " ")
        return shape_str
    
    #cv format nhwc
    def to_cv_transpose(self):
        return list(self._layout).index(dimension.N), list(self._layout).index(dimension.H), list(self._layout).index(dimension.W), list(self._layout).index(dimension.C)

    @property
    def batch(self):
        return self._layout[dimension.N]

    @property
    def height(self):
        return self._layout[dimension.H]

    @property
    def weight(self):
        return self._layout[dimension.W]
    
    @property
    def channel(self):
        return self._layout[dimension.C]

    @property
    def layout(self):
        return self._layout
    
    @property
    def shape_value(self):
        return self.shape



type_mapping = {
    "int": np.int32,
    "unit8": np.uint8,
    "float": np.single,
    "double": np.double,
    "half": np.half
}

def load_image(path : str, img_shape : shape, precision_str):
    with open(path, 'rb') as f:
        img_bytes = f.read()
    if precision_str not in type_mapping:
        raise Exception("input precision error ", precision_str)
    img_array = np.frombuffer(img_bytes, dtype=type_mapping[precision_str])
    return img_array.reshape(img_shape.shape_value).astype(np.uint8).transpose(img_shape.to_cv_transpose())

def show_img(path : str, img_array : np.array, bash : int):
    for i in range(0,bash):
        cv2.imshow(path, img_array[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(argv):
    layout_str = 'nhwc'
    shape_str = '1*1080*1920*3'
    file = ''
    precision_str = 'uint8'
    opts, args = getopt.getopt(argv,"hf:l:s:p:",["filepath=","layout=","shape=","precision="])
    for opt, arg in opts:
        if opt == '-h':
            print ('image_ready.py -f <filepath> -i <layout: nchw> -s <shape: 2*3*768*1024> -p <precision: float/double/int/half>')
            sys.exit()
        elif opt in ("-l", "--layout"):
            layout_str = arg
        elif opt in ("-s", "--shape"):
            shape_str = arg
        elif opt in ("-f", "--file"):
            file = arg
        elif opt in ("-p", "--precision"):
            precision_str = arg
    img_shape = shape(shape_str, layout_str)
    print("image shape: ", str(img_shape))
    img = load_image(file, img_shape, precision_str)
    show_img(file, img, img_shape.batch)

if __name__ == "__main__":
   main(sys.argv[1:])