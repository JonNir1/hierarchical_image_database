# Hierarchical Image Dataset - Source Manifest
This directory contains the full 725-image dataset, organized into a hierarchical structure 
based on the directory tree. Each image is stored in its respective subdirectory, which reflects
its semantic category.<br>
This dataset is a subset of the 770-image dataset curated by Auerbach-Asch et al. (2023). Images 
were dropped due to failure to locate their source or because they belonged to an overly-populated
category.<br>
The images originate from previously published datasets or publicly available sources, as documented 
in the `manifest.csv` file.

## Pre- and Post-SHINE Variants
The dataset includes two variants of each image:
- **Pre-SHINE**: The original images, after background removal and resizing to 170x170 pixels.
- **Post-SHINE**: The images after being processed by the SHINE_color toolbox (Dal Ben, R., 2023)
  for luminance and color normalization.

## Manifest
The `manifest.csv` file provides a comprehensive listing of all images in the dataset, including 
a reference to their source dataset. Each row corresponds to a single image and includes the 
following columns:

| column | description                                                                                                                                                                                                                                              |
|---|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `curated_path` | Relative path inside `pre_shine/` and `post_shine/`, Backslash-separated.<br/>Encodes the hierarchical taxonomy (animacy / kingdom / body-part / sub-category).                                                                                          |
| `curated_filename` | Image's filename (basename of `curated_path`), with semantic classification of the image (e.g., `chick1.png`, `business_man2.png`).                                                                                                                      |
| `category` | Paper category code as reported in Auerbach-Asch et al. (2023).<br/>One of: `HF` (Human Faces), `AF` (Animal Faces), `HB` (Human Bodies), `AB` (Animal Bodies), `NO` (Natural Objects), `HO` (Handmade Objects).                                         |
| `source_dataset` | Human-readable label of the dataset/source of origin. See **Source datasets** below for the full citation of each label.                                                                                                                                 |
| `source_filename_or_url` | If `source_dataset` is one of the named scientific/lab datasets, this is the original filename in that dataset's canonical distribution. For `online source`, this is the URL where the image was found (typically the page URL, not the raw image URL). |
| `manual_match_validation` | `V` for every row — each match was visually confirmed by an author. No row appears in this manifest that has not been eyeballed against its source.                                                                                                      |


## Source datasets

Counts in parentheses are the number of curated images attributed to each source.

### Kiani et al., 2007 (391)
1,084-image colour photograph stimulus set introduced for inferotemporal
cortex experiments in macaques. We utilized 391 of the 1,084 images, keeping 
the canonical 8-digit numeric filenames (`NNNNNNNN.bmp`). <br>
- Public archive: <https://www.cns.nyu.edu/kianilab/datasets/Kiani2007/Kiani_ImageSet.tar.gz>
- Licence: **<TODO: check>**

### Face Place (102)
Tarr Lab face database (Release 3.0). 250 × 250 colour JPEGs of over 200
individuals across five ethnicities, each with multiple views, expressions, 
and disguises (Righi et al., 2012). Face images courtesy of Michael J. Tarr,
Carnegie Mellon University, http://www.tarrlab.org/. Funding provided by 
NSF award 0339122.<br>
All 102 HF (Human Faces) curated images come from this set. 
- Public archive: <https://sites.google.com/andrew.cmu.edu/tarrlab/stimuli>
- Licence: CC BY-NC-SA 3.0

### Grootswagers et al., 2019 (98)
200 colour-object stimuli plus 16 targets (8 boats + 8 geometric stars),
originally sourced by the authors from the free image-hosting site
pngimg.com and organized into a 3-level hierarchy (animacy → category →
object exemplar).
- Public archive (OSF): <https://osf.io/a7knv/>
- License: **<TODO: check>**

### Online Source (57)
A heterogeneous collection of images sourced from online image repositories, 
such as Wikipedia, Wikimedia Commons, and pngimg.com. The specific source URL
is provided for each image in the `source_filename_or_url` column of the manifest.
- Public archive: N/A (please refer to the source URLs in the manifest)
- License: Varies by image; please refer to the source URLs for licensing information.

### HUJI Flowers (45)
Images adapted from the [Israeli Flower Database](https://www.cs.huji.ac.il/~daphna/IsraeliFlowers/flower_classification.html), 
which contains over 3,000 images of 115 Israeli wildflower species. The 45 curated 
images are background-removed versions of the original images.
- Public archive: <https://www.cs.huji.ac.il/~daphna/IsraeliFlowers/download_israel_db.html>
- License: non-commercial research and educational purposes only

### ANU Butterflies (21)
Twenty-one butterfly/moth images from the Australian National University's online 
_Butterflies and Moths Image Archive_.
- Public archive: <http://artserve.anu.edu.au/new/australia/canberra/CSIRO/butterflies-moths/index1.html>
- License: **<TODO: check>**

### Auerbach-Asch & Deouell (2024) (11)
Eleven analog-watch images that were also used in the Auerbach-Asch & Deouell (2024) study. 
Please refer to that paper's methods section for details on the source and licensing of these images.


## Category Taxonomy

### Top-Level Categories
The six top-level categories from Auerbach-Asch et al. (2023) map to the directory tree as follows:

| category         | code | directory subtree      |   n |
|------------------|------|------------------------|----:|
| Human Faces      | HF   | `animate/human/face/`  | 102 |
| Animal Body      | AB   | `animate/animal/body/` |  81 |
| Animal Faces     | AF   | `animate/animal/face/` |  53 |
| Human Body       | HB   | `animate/human/body/`  |  62 |
| Natural Objects  | NO   | `inanimate/natural/`   | 160 |
| Handmade Objects | HO   | `inanimate/handmade/`  | 267 |

### Full Taxonomy
ASCII representation of the directory structure of the curated dataset.<br>
Numbers in parentheses count the .png files at or beneath each directory.<br>
Leaf-level individual stimulus files (e.g., `chick1.png`) are not shown.

```
pre_shine (725)
├── animate (298)
│   ├── animal (134)
│   │   ├── body (81)
│   │   │   ├── bird (17)
│   │   │   ├── butterfly (31)
│   │   │   ├── fish (3)
│   │   │   └── mammal (30)
│   │   │       ├── canine (8)
│   │   │       ├── feline (4)
│   │   │       ├── hooved (10)
│   │   │       ├── marsupial (1)
│   │   │       ├── primate (2)
│   │   │       └── rodent (5)
│   │   └── face (53)
│   │       ├── bird (10)
│   │       └── mammal (43)
│   │           ├── bear (1)
│   │           ├── canine (4)
│   │           ├── feline (6)
│   │           ├── hooved (12)
│   │           ├── primate (19)
│   │           └── rodent (1)
│   └── human (164)
│       ├── body (62)
│       │   ├── child (5)
│       │   ├── clown (3)
│       │   ├── hand (27)
│       │   ├── man (17)
│       │   └── woman (10)
│       └── face (102)
│           ├── asian (23)
│           ├── black (16)
│           └── caucasian (63)
└── inanimate (427)
    ├── handmade (267)
    │   ├── ball (24)
    │   ├── clothing (34)
    │   ├── electric (18)
    │   ├── food (6)
    │   ├── furniture (43)
    │   ├── kitchen (22)
    │   ├── other (26)
    │   ├── tool (64)
    │   └── vehicle (30)
    │       └── car (20)
    └── natural (160)
        ├── flower (71)
        │   ├── blue (3)
        │   ├── pink-purple (18)
        │   ├── red (14)
        │   ├── white (16)
        │   └── yellow (20)
        ├── food (60)
        └── other (29)
```

### References
```
@article{auerbach2024beyond,
  title={Beyond Stimulus Onset: Ongoing Fixations Within an Object Do Not Re-evoke Category Representations During Free-Viewing},
  author={Auerbach-Asch, Carmel Ruth and Deouell, Leon Y},
  journal={bioRxiv},
  pages={2024--12},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}

@article{carmel2023decoding,
  title={Decoding object categories from EEG during free viewing reveals early information evolution compared to passive viewing},
  author={Auerbach-Asch, Carmel R. and Vishne, Gal and Wertheimer, Oded and Deouell, Leon Y.},
  journal={BioRxiv},
  pages={2023--06},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}

@article{dal2023shine_color,
  title={SHINE\_color: Controlling low-level properties of colorful images},
  author={Dal Ben, Rodrigo},
  journal={MethodsX},
  volume={11},
  pages={102377},
  year={2023},
  publisher={Elsevier}
}

@article{kiani2007object,
  title={Object category structure in response patterns of neuronal population in monkey inferior temporal cortex},
  author={Kiani, Roozbeh and Esteky, Hossein and Mirpour, Koorosh and Tanaka, Keiji},
  journal={Journal of neurophysiology},
  volume={97},
  number={6},
  pages={4296--4309},
  year={2007},
  publisher={American Physiological Society}
}

@article{grootswagers2019representational,
  title={The representational dynamics of visual objects in rapid serial visual processing streams},
  author={Grootswagers, Tijl and Robinson, Amanda K and Carlson, Thomas A},
  journal={NeuroImage},
  volume={188},
  pages={668--679},
  year={2019},
  publisher={Elsevier}
}

@article{righi2012recognizing,
  title={Recognizing disguised faces},
  author={Righi, Giulia and Peissig, Jessie J and Tarr, Michael J},
  journal={Visual Cognition},
  volume={20},
  number={2},
  pages={143--169},
  year={2012},
  publisher={Taylor \& Francis}
}
```
