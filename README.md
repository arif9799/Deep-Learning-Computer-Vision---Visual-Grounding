# Deep-Learning---Visual-Grounding

This Repository contains all about the basic implementation of visual grounding using transformers in PyTorch Library.
The details of the implementation, navigation of the repository, different modes of data used and complete walkthrough of the code 
will soon be updated. 

## About Visual Grounding:

- Visual grounding intents to locate an object instance in an image referred to by a natural language query. 
It is comparatively different from object detection, in which the detected objects belong to a set
of pre-defined classes whereas in visual grounding requires an understanding of the query, image and
context between them. In visual grounding, the referred object is specified by pieces of information
in the language expression. The information may include object categories, appearance, and visual
relation,etc. Visual grounding is mainly applied to bridge the gap between visual perception and
linguistic expression in the field of human computer interaction.

- Historically visual grounding was treated as a form of text-based image retrieval, where object
detectors were used to identify regions and regions are then ranked based on similarity with query.
The two-staged and one-staged methods performance mainly depended on the pre-predicted proposals
or pre-defined anchor boxes such methods may be less flexible for capturing contexts mentioned in
linguistic query and visual image. and was computationally expensive to generate all possible regions.
This expense in computation could be dealt by removing proposal search and fusing independent
textual context and visual features to find exact regions.

- The recent boom of transformers in natural language processing and computer vision has
shown promising results and motivated us to utilize the power of self-attention in transformers
to develop a visual grounding for the given query. The visual features are extracted using visual
transformer(ViT) and the textual features are extracted using BERT. The baseline model
would be a concatenation of linguistic and vision features. A regression would be performed on the
integrated features to get the referred object and it is evaluated based on using Accuracy@0.5. The
aim is to get better Accuracy@0.5 compared to baseline model. To achieve this we integrate the
liguistic and vission features Transformers with self-attention heads. The model is trained on a
part of Ficklr30k Entities data. Regression will be performed on the advance embeddings obtained
to get coordinates of bounding box around the referred object.



## Different Interesting Architectures attempted.

The inputs or the embeddings for Images & Text used to train the models were precalculated using Vision Transformer & BERT respectively

- Baseline Model.
  - The Text and Image Embeddings were concatenated and feed forward through a MultiLayer Perceptron head to predict the 4 coordinates of a Bounding Box
  - <img src="https://user-images.githubusercontent.com/93501171/234690140-0e2dca2d-0a75-4234-8ee1-3437097d8fb9.png" data-canonical-src="https://user-images.githubusercontent.com/93501171/234690140-0e2dca2d-0a75-4234-8ee1-3437097d8fb9.png" width="400" height="400" />

- Encoder Decoder Model
  The Encoder Decoder consists of a custom encoder also termed as Grounding Encoder by the Original Authors of the paper Visual grounding with Transformers.
  - The Grounding Encoder consists of N Layers, and each layer has 2 Interconnected Grounding Encoder Cells, each with 8 Multi-Attention heads
  - The first Grounding Encoder Cell takes as input the Text Embeddings from BERT and produces the encoded text.
  - The second Grounding Encoder Cell takes as input the Image Embeddings from ViT along with encoded text from first Encoder Cell, fuses them to generate text guided encoded image
  - The encoded text output from the first encoder cell serves as input to self-attention head in Decoder
  - The text guided encoded image from the second encoder cell serves as input to cross-attention head in Decoder
  - The Output of the Decoder passes through MultiLayer Perceptron to predict the 4 coordinates of a Bounding Box
  - <img src="https://user-images.githubusercontent.com/93501171/234698400-19a96046-cec6-4b97-95c9-3ccdace4f8ff.png" data-canonical-src="https://user-images.githubusercontent.com/93501171/234698400-19a96046-cec6-4b97-95c9-3ccdace4f8ff.png" width="600" height="600" />
  
  
- Text Encoder Cell and Decoder Model
  This Model has its Grounding Encoder modified to only contain text encoder cell
  - The Grounding Encoder consists of N Layers, and each layer has a text encoder cell, each with 8 Multi-Attention heads
  - The text Encoder Cell in Grounding Encoder takes as input the Text Embeddings from BERT and produces the encoded text
  - The encoded text output from the text encoder cell serves as input to self-attention head in Decoder
  - The image embeddings from ViT are served as input to cross-attention head in Decoder
  - The Output of the Decoder passes through MultiLayer Perceptron to predict the 4 coordinates of a Bounding Box
  - <img src="https://user-images.githubusercontent.com/93501171/234698568-3d0b1832-4897-43b9-99c8-c32734465ff8.png" data-canonical-src="https://user-images.githubusercontent.com/93501171/234698568-3d0b1832-4897-43b9-99c8-c32734465ff8.png" width="400" height="600" />
  
  
- Visual Encoder Cell and Decoder Model
  This Model has its Grounding Encoder modified to only contain text encoder cell
  - The Grounding Encoder consists of N Layers, and each layer has a visual encoder cell, each with 8 Multi-Attention heads
  - The visual Encoder Cell in Grounding Encoder takes as input the Image Embeddings from ViT and produces the encoded image
  - The text embeddings from BERT serves as input to self-attention head in Decoder
  - The encoded image from visual encoder cell serves as input to cross-attention head in Decoder
  - The Output of the Decoder passes through MultiLayer Perceptron to predict the 4 coordinates of a Bounding Box
  - <img src="https://user-images.githubusercontent.com/93501171/234698990-48c3c982-1d12-474a-84b8-214987c33bed.png" data-canonical-src="https://user-images.githubusercontent.com/93501171/234698990-48c3c982-1d12-474a-84b8-214987c33bed.png" width="400" height="600" />
  
  
- Decoder Only Model
  This Model has only decoder and directly takes the text and image embeddings as inputs from BERT and ViT
  This Model was attemtpted to produce light-weight model
  - The self-attention head in decoder takes as input text embeddings from BERT
  - The cross-attention head in decoder takes as input image embeddins from ViT
  - The Output of the Decoder passes through MultiLayer Perceptron to predict the 4 coordinates of a Bounding Box
  - <img src="https://user-images.githubusercontent.com/93501171/234699180-26141d8e-f4b4-495d-89e9-94f672f95b2f.png" data-canonical-src="https://user-images.githubusercontent.com/93501171/234699180-26141d8e-f4b4-495d-89e9-94f672f95b2f.png" width="600" height="400" />

## Results on Flickr30k entities with Single Bounding Box Prediction.

|     Model                               | Validation Accuracy  | Test Accuracy |
|:-------------------                     |:---------------      |:------------  |
| Baseline Model                          | 68.91                | 88.51         |
| Encoder Decoder Model                   | 78.27                | 92.67         |
| Image Encoder Cell with Decoder Model   | 64.03                | 69.22         |
| Text Encoder Cell with Decoder Model    | 72.03                | 70.38         |
| Decoder Only Model                      | 64.01                | 71.61         |





