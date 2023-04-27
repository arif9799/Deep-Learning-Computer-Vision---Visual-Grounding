######################################################################################################################
#                                                                                                                    #
#                                        Give Image name and extract Absolute Paths                                  #
#                                                                                                                    #
######################################################################################################################

def get_Paths(image_name, _paths):
    
    _sentences_path = _paths['_sentences_path']
    _annotations_path = _paths['_annotations_path']
    _image_folder_path = _paths['_image_folder_path']
    
    """
    Inputs : 
        image_name: image file name (unique id)
        
    Outputs:
        s : Image Sentence and Phrase Text File Absolute Paths
        a : Image Bounding Boxes XML file absolute path
        ab : Original Image File jpg path
    
    """
    s = os.path.join(_sentences_path, image_name + '.txt')
    a = os.path.join(_annotations_path, image_name + '.xml')
    ab = os.path.join(_image_folder_path, image_name + '.jpg')
    
    return s, a, ab

"""***************************************************************************************************************"""






######################################################################################################################
#                                                                                                                    #
#                          Load predefined Train, Validation & Test Split as per original authors                    #
#                                                                                                                    #
######################################################################################################################

import os
def load_Splits(file_path : str):
    """
    inputs:
        file_path: absolute path to the txt files that contain the images belonging to corresponding split,
            for eg, train.txt contains all image ids that belong to train split
                    val.txt contains all images that belong to validation split
                    test.txt contains all images that belong to test split
    outputs:
        image_names: list of image names (list of strings)
    """
    
    image_names = None
    with open(file_path) as fp:
        image_names = fp.read().splitlines()
        
    return image_names
"""***************************************************************************************************************"""







######################################################################################################################
#                                                                                                                    #
#                 Pass Sentences(with Phrases) and Annotations & it'll return map (Phrase to Bbox)                   #
#                                                                                                                    #
######################################################################################################################

def phrase_Id_to_Bbox(sentences, annotations):
    """
    Inputs:
        sentences: the extracted sentences from getSentences function, it contains multiple sentences meaning the same,
                   and then broken down into phrases, with phrase_ids
        annotations: dictionary of dictionaries, out of which, the important one is dictionary of phrases in it.
                     From the dictionary of phrases (phrase_id and corresponding Bounding Boxes), we extract and create
                     a dictionary of bounding boxes for each image
                     
                     
    Outputs:
        _phrase_id_to_bbox: A dictionary with image id as keys, and a dictionary of (phrase_ids & bounding boxes) as values
        
    """
    _phrase_id_to_bbox = {}
    existing_phrase_ids = annotations['boxes'].keys()
    for sentence in sentences:
        for phrase in sentence['phrases']:
            if phrase['phrase_id'] in existing_phrase_ids:
                _phrase_id_to_bbox[phrase['phrase_id']] = annotations['boxes'][phrase['phrase_id']]
    
    return _phrase_id_to_bbox

"""***************************************************************************************************************"""






######################################################################################################################
#                                                                                                                    #
#             Pass Sentences(with Phrases) and Annotations & it'll return map (Phrase_id to Phrase)                  #
#                                                                                                                    #
######################################################################################################################

def phrase_Id_to_Phrases(sentences, annotations):
    """
    Inputs:
        sentences: the extracted sentences from getSentences function, it contains multiple sentences meaning the same,
                   and then broken down into phrases, with phrase_ids
        annotations: dictionary of dictionaries, out of which, the important one is dictionary of phrases in it.
                     From the dictionary of phrases (phrase_id and corresponding Bounding Boxes), we extract and create
                     a dictionary of bounding boxes for each image
                     
                     
    Outputs:
        _phrase_id_to_phrases: A dictionary with image id as keys, and a dictionary of (phrase_ids & corresponding phrases) as values
        
    """
    _phrase_id_to_phrases = {}
    for sentence in sentences:
        for phrase in sentence['phrases']:
            if phrase['phrase_id'] not in list(_phrase_id_to_phrases.keys()):
                _phrase_id_to_phrases[phrase['phrase_id']] = set()
            _phrase_id_to_phrases[phrase['phrase_id']].add(phrase['phrase'])
    return _phrase_id_to_phrases

"""***************************************************************************************************************"""








######################################################################################################################
#                                                                                                                    #
#                                            Plot the Patches of the Phrases                                         #
#                                                                                                                    #
######################################################################################################################

from tqdm import tqdm
from PIL import Image
import numpy as np


import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patches as patches # for bounding boxes
import matplotlib.colors as mcolors
plt.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['figure.dpi'] = 100

def plot_patches(_Image_id, _Bboxes, _Phrs, _paths, _OnlyBbox = False, _Single_Patch = False, _COLORS = mcolors.CSS4_COLORS):
    
    
    """
    Fetch the Image
    """
    _,_,_img_path = get_Paths(_Image_id, _paths)
    _Image = Image.open(_img_path)
    fig, ax = plt.subplots()
    _COLORS = list(_COLORS.keys())
    
    """
    _Blank is just when you want the bounding box and not the image in the background.
    """
    _Blank = np.ones((_Image.height, _Image.width), dtype=float)
    _Blank.fill(255)
    
    
    if _Single_Patch:
        color = _COLORS.pop()
        x = float(_Bboxes[0])
        y = float(_Bboxes[1])
        w = float(_Bboxes[2]) - x
        h = float(_Bboxes[3]) - y

        plt.text(x+1, y-5, _Phrs, fontdict = {'fontsize':8.0, 'fontweight':'medium', 'color':'white', 'backgroundcolor': 'black'})
        bb = patches.Rectangle((x,y), w, h, linewidth = 2, edgecolor = color, facecolor = 'None')
        ax.add_patch(bb)
        ax.imshow(_Image)
        
        
    else:
    
        """
        Get the bounding Boxes & Phrases for that
        """
        BoundingBoxes = _Bboxes[_Image_id]
        Phrases = _Phrs[_Image_id]

        for phrase_id, phrases in Phrases.items():
            if phrase_id not in list(BoundingBoxes.keys()):
                continue
            bboxes = BoundingBoxes[phrase_id]
            name = str(phrases)
            for _Bbox in bboxes:

                color = _COLORS.pop()
                """
                [ x_min, y_min, x_max, y_max ] -----> [ _Bbox[0], _Bbox[1], _Bbox[2], _Bbox[3] ]
                """
                x = float(_Bbox[0])
                y = float(_Bbox[1])
                w = float(_Bbox[2]) - x
                h = float(_Bbox[3]) - y

                if not _OnlyBbox:
                    plt.text(x+1, y-5, name, fontdict = {'fontsize':18.0, 'fontweight':'medium', 'color':'white', 'backgroundcolor': 'black'})
                bb = patches.Rectangle((x,y), w, h, linewidth = 2, edgecolor = color, facecolor = 'None')
                ax.add_patch(bb)


        if _OnlyBbox: 
            ax.imshow(_Blank, cmap= 'gray',vmin= 0,vmax= 1)
        else:
            ax.imshow(_Image)
            
            
            
            
            
    plt.show()
    
"""***************************************************************************************************************"""





######################################################################################################################
#                                                                                                                    #
#                                                 Prepare the DataFrame                                              #
#                                                                                                                    #
######################################################################################################################

import pandas as pd
def prepare_DataFrame(Phrase_Dict, Bbox_Dict):
    Final_DF = pd.DataFrame()
    for Image_Id in tqdm(Phrase_Dict.keys()):
        
        Phrase_DF = pd.DataFrame.from_dict(Phrase_Dict[Image_Id], orient = 'index')
        Phrase_DF = pd.DataFrame(Phrase_DF.stack(level=0)).reset_index().drop('level_1', axis = 1)

        Bbox_DF = pd.DataFrame.from_dict(Bbox_Dict[Image_Id], orient = 'index')
        Bbox_DF = pd.DataFrame(Bbox_DF.stack(level=0)).reset_index().drop('level_1', axis = 1)

        Merged_DF = pd.merge(Phrase_DF, Bbox_DF, on = 'level_0', how='inner')
        Merged_DF['Image_Id'] = Image_Id

        Final_DF = pd.concat([Final_DF, Merged_DF], axis = 0)

    Final_DF = Final_DF.rename(columns = {'level_0' : 'Phrase_Id', '0_x': 'Phrase', '0_y':'Bounding_Box'})
    Final_DF = Final_DF[['Image_Id', 'Phrase_Id', 'Phrase', 'Bounding_Box']]
    Final_DF.reset_index(drop = True, inplace = True)
    Final_DF[['x_min', 'y_min', 'x_max', 'y_max']] = Final_DF.Bounding_Box.to_list()
    
    
    return Final_DF



"""***************************************************************************************************************"""





######################################################################################################################
#                                                                                                                    #
#                            Get just Image sizes & not unnecessrily all the images                                  #
#                                                                                                                    #
######################################################################################################################


import imagesize
from tqdm.notebook import tqdm


def get_Image_Sizes(List_Images, _Paths_Images):
    aspect_dct = {}
    for img in tqdm(List_Images):
        _,_,path = get_Paths(img, _Paths_Images)
        aspect_dct[str(img)] = imagesize.get(path)

    aspect_df = pd.DataFrame.from_dict([aspect_dct]).T.reset_index()
    aspect_df.set_axis(['FileName', 'Size'], axis = 'columns', inplace=True)
    aspect_df[['Width', 'Height']] = pd.DataFrame(aspect_df["Size"].tolist(), index = aspect_df.index)
    aspect_df['aspect_ratio'] = round(aspect_df['Width']/aspect_df['Height'],2)
    
    return aspect_df

