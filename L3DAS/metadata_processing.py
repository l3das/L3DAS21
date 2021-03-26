import pandas as pd
import numpy as np
import math

def get_label(path,frame_len,file_size,sample_rate,classes_,num_frames):
    class_vec=[]
    loc_vec=[]
    num_classes=len(classes_)
    for k in range(num_frames):
        class_vec.append([None]*num_classes*3)
        loc_vec.append([None]*(num_classes*9))

    class_dict={}
    df=pd.read_csv(path,index_col=False)
    classes_names=classes_
    #classes_names=df['sound_event_recording'].unique()
    #classes_names=classes_names.tolist()
    data_for_class=[]
    for class_name in classes_names:
        class_dict[class_name]=[]
    for class_name in classes_names:
        #data_for_class=df.loc[df['sound_event_recording'] == class_name]
        data_for_class=df.loc[df['Class'] == class_name]
        data_for_class=data_for_class.values.tolist()
        for data in data_for_class:
            class_dict[data[3]].append([data[1],data[2],data[4],data[5],data[6]])
    
    for j, i in enumerate(np.arange(frame_len,file_size+frame_len,frame_len)):
        start=i-frame_len
        end=i
        for clas in class_dict:
            data_list=class_dict[clas]
            for data in data_list:
                is_in=False
                start_audio=data[0]
                end_audio=data[1]
                x_audio=data[2]
                y_audio=data[3]
                z_audio=data[4]
                if start_audio<start and end_audio>end:
                    is_in=True
                elif end_audio>start and end_audio<end:
                    is_in=True
                elif start_audio>start and start_audio<end:
                    is_in=True
                if is_in:
                    ind_class=classes_names.index(clas)*3
                    if class_vec[j][ind_class]==None:
                        class_vec[j][ind_class]=1
                    elif class_vec[j][ind_class]!=None and class_vec[j][ind_class+1]==None:
                        class_vec[j][ind_class+1]=1
                    else:
                        class_vec[j][ind_class+2]=1
                    if loc_vec[j][ind_class*3]==None:
                        loc_vec[j][0+int(ind_class*3)]=x_audio
                        loc_vec[j][1+int(ind_class*3)]=y_audio
                        loc_vec[j][2+int(ind_class*3)]=z_audio
                    elif loc_vec[j][ind_class*3]!=None and loc_vec[j][ind_class+1]==None:
                        loc_vec[j][3+int(ind_class*3)]=x_audio
                        loc_vec[j][4+int(ind_class*3)]=y_audio
                        loc_vec[j][5+int(ind_class*3)]=z_audio
                    else:
                        loc_vec[j][6+int(ind_class*3)]=x_audio
                        loc_vec[j][7+int(ind_class*3)]=y_audio
                        loc_vec[j][8+int(ind_class*3)]=z_audio
        for i in range(len(loc_vec[j])):
            if loc_vec[j][i]==None:
                loc_vec[j][i]=0
        for i in range(len(class_vec[j])):
            if class_vec[j][i]==None:
                class_vec[j][i]=0
    return (class_vec), (loc_vec)
                



    
    