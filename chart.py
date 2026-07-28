#! /usr/bin/env python3
import matplotlib.pyplot as plt
import sys
import os

path = "images/"

def draw_charts(crop_dict):
    fig,plots=plt.subplots(ncols=2,nrows=1, figsize= (20, 8))
    labels= crop_dict.keys()
    sizes= crop_dict.values()
    plots[0].pie(sizes,labels= labels,autopct="%1.1f%%")
    plots[1].bar(labels,sizes,color=["orange","pink","yellow","red"])
    plt.savefig("chart.png")
    # plt.show()

def get_labels_and_sizes(crop_name):
    crop = {}
    for folder in os.listdir(path):
        if folder.find(crop_name)==0:
            crop[folder] =  len(os.listdir(os.path.join(path,folder)))
    if len(crop)!= 0  :
        draw_charts(crop)




if len(sys.argv) != 2:
    print("you should provide at least one folder name")
else :
    crop_name = sys.argv[1]
    get_labels_and_sizes(crop_name)