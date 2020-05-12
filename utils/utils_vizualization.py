import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from PIL import Image
from itertools import cycle

cycol = cycle([(255, 0, 0), (255, 128, 0), (255, 153, 255), (0, 255, 0), (0, 0, 255), (128, 0, 255), (128, 255, 0), (0, 128, 255)])

def plot_points(im_path, points, labels, out_path=None, save=False, show=False):
    plt.rcParams["figure.figsize"] = (20,20)
    im = np.array(Image.open(im_path), dtype=np.uint8)
    height, width = im.shape[0:2]
    print(str(height)+" "+str(width))
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    for i in range(len(points)):
        x, y = points[i]
        plt.scatter(x, y, c='green', marker='o')
    if save:
        plt.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)

def plot_boxes_for_points(img_name, points, labels, outs, out_path, save=False, show=False):
    plt.rcParams["figure.figsize"] = (20,20)
    im = np.array(Image.open(img_name), dtype=np.uint8)
    height, width = im.shape[0:2]
    print(str(height)+" "+str(width))
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    b = []
    print(img_name + str(points)+str(labels))
    found = False
    for i in range(len(points)):
        x, y = points[i]
        c = next(cycol)
        for j in range(int(outs[1])):
            ind = 2+6*j
            x_min=int(float(outs[ind+2])*width)
            y_min=int(float(outs[ind+3])*height)
            w=int(float(outs[ind+4])*width)
            h=int(float(outs[ind+5])*height)
            if (is_pt_in_box([x, y], [x_min, y_min, x_min+w, y_min+h])):
                rect = patches.Rectangle((x_min,y_min),w,h,linewidth=3,edgecolor=c,facecolor='none')
                ax.add_patch(rect)
                b.append([x_min, y_min, w, h])
                plt.scatter(x, y, c=c, marker='o')
                ax.text(x+2.0, y+2.0, labels[i], fontsize=24, color=c)
                print(str(x)+" "+str(y)+" "+labels[i])
                found = True
                break
    if found:
        if save:
            plt.savefig(out_path)
        if show:
            plt.show()
    plt.close(fig)
    return b

def is_pt_in_box(pt, box):
    x, y = pt
    x_min, y_min, x_max, y_max = box
    if x_min <= x and x <= x_max and y_min <= y and y <= y_max:
        return True
    else:
        return False

def plot_cv(im_path, outputs, out_path=None, save=False, show=False):
    img = cv2.imread(im_path)
    height, width = img.shape[0], img.shape[1]
    size = height*width/(1000*1000)
    label_size = int(min(size, 10))
    for i in range(int(outputs[1])):
        ind = 2+6*i
        x=int(float(outputs[ind+2])*width)
        y=int(float(outputs[ind+3])*height)
        w=int(float(outputs[ind+4])*width)
        h=int(float(outputs[ind+5])*height)
        x_min, y_min, x_max, y_max = x, y, x+w, y+h
        label = outputs[ind]
        label = label.encode('ascii','ignore').decode('utf-8')
        label = str(i+1)+"_"+label
        c = next(cycol)
        cv2.rectangle(img,(int(x_min),int(y_min)),(int(x_max),int(y_max)),c, label_size)
        x_label = x_min+5
        if y_min-10 < 0:
            y_label = y_min+int(4*label_size)
        else:
            y_label = y_min-int(4*label_size)
        cv2.putText(img, label, (x_label, y_label), cv2.FONT_HERSHEY_SIMPLEX, label_size/3, c, label_size)
    cv2.imwrite(out_path, img)

def plot_matplot(im_path, outputs, out_path=None, save=False, show=False):
    plt.rcParams["figure.figsize"] = (10,10)
    im = np.array(Image.open(im_path), dtype=np.uint8)
    height, width = im.shape[0:2]
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    ax.text(0.0, 0.0, outputs[1], fontsize=60, color='b')
    for i in range(int(outputs[1])):
        ind = 2+6*i
        x=int(float(outputs[ind+2])*width)
        y=int(float(outputs[ind+3])*height)
        w=int(float(outputs[ind+4])*width)
        h=int(float(outputs[ind+5])*height)
        x_min, y_min, x_max, y_max = x, y, x+w, y+h
        if x_min < 0.0:
            x_min = 0.0
        if y_min < 0.0:
            y_min = 0.0
        if x_max > 1.0:
            x_max = 1.0
        if y_max > 1.0:
            y_max = 1.0
        label = str(outputs[ind])
        label = label+"_"+str(round(float(outputs[ind+1]), 2))
        c = next(cycol)
        rect = patches.Rectangle((x,y),w,h,linewidth=3,edgecolor=c,facecolor='none')
        ax.text(x+w/2.0, y+2.0, label, fontsize=24, color=c)
        ax.add_patch(rect)
    if save:
        plt.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)