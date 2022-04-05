# Dataset Readme
The code currently supports the NCars, the NCaltech101 and the Gen1 event datasets.

### Recognition Task
The Neuromorphic N-Caltech101 dataset contains event streams recorded with a real event camera 
representing 101 object categories in 8,246 event sequences. each 300 ms long, mirroring the well-known 
Caltech101 dataset for images. The N-Cars dataset has real events, assigned to either a car or the
background. It has 24, 029 event sequences, each being 100 ms long. 

### Detection Task
The N-Caltech101 dataset and contains only one bounding box per sample, it contains 101 classes, making 
it a difficult classification task. By contrast, Gen1 targets an automotive scenario in an urban environment with
annotated pedestrians and cars. With 228,123 bounding boxes for cars and 27,658 for pedestrians, the Gen1 dataset
is much larger.

## Pre-Processing
```
CUDA_VISIBLE_DEVICES=X python3 aegnn/scripts/preprocessing.py --dataset dataset --num-workers X
# dataset = ["ncars", "ncaltech101", "gen1"]
```

## File Structure
For a valid pre-processing the code expects the same file structure over all input dataset. It expects a `training`, 
`validtion` and `test` directory in the dataset's folder, each containing the raw data files in the respective set. In 
the subset's directories the data are either sorted by their class label or sequence id. 

### NCaltech101
```
ncaltech101/
    training/
        accordion/
            image_0001.bin
            ...
        anchor/
        ... 
    validation/
        ...
    test/
        ...
```

### NCars
```
ncars/
    training/
        sequence_00001/
            events.txt
            is_car.txt
        sequence_00001/
        ... 
    validation/
        ...
    test/
        ...
```

### Gen1
```
gen1/
    training/
        17-04-13_12-15-24_854500000_914500000_td.dat
        17-04-13_12-15-24_854500000_914500000_bbox.npy
        ... 
    validation/
        ...
    test/
        ...
```

