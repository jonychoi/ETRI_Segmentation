import numpy as np

_classesNames = [
    "person", 
    "bicycle", 
    "car", 
    "train", 
    "boat", 
    "bench", 
    "bird", 
    "cat", 
    "dog",
    "horse", 
    "sheep", 
    "cow", 
    "elephant", 
    "bear",
    "zebra", 
    "giraffe", 
    "hat", 
    "umbrella", 
    "bottle", 
    "wine_glass", 
    "cup", 
    "fork", 
    "knife", 
    "spoon", 
    "bowl", 
    "banana", 
    "apple", 
    "orange", 
    "carrot", 
    "cake", 
    "chair(bench,couch)",
    "potted_plant", 
    "bed", 
    "table(dinning_table,desk)", 
    "book", 
    "clock", 
    "vase", 
    "blanket", 
    "bridge", 
    "building-other",
    "ceiling-other",
    "clouds", 
    "curtain", 
    "dirt(mud)",
    "floor-other(carpet,mat,rug,platform)",
    "flower", 
    "food-other",
    "fruit", 
    "furniture-other(shelf,"
    "grass(moss)",
    "gravel", 
    "ground-other(leaves)",
    "hill", 
    "house", 
    "light", 
    "mountain", 
    "pavement", 
    "pillow", 
    "plant-other",
    "railroad", 
    "road", 
    "rock", 
    "sand", 
    "sea", 
    "sky-other",
    "snow", 
    "stairs", 
    "grapes", 
    "towel", 
    "tree", 
    "vegetable", 
    "wall-other",
    "water-other",
    "window-other",
    "potato", 
    "pear", 
    "peach", 
    "bread", 
    "fish", 
]

def classNames():
    return _classNames

union = [
    {'id': 1, 'name': 'person', 'counts': 374}, 
    {'id': 0, 'name': 'background', 'counts': 321}, 
    {'id': 65, 'name': 'sky-other', 'counts': 262}, 
    {'id': 70, 'name': 'tree', 'counts': 224}, 
    {'id': 72, 'name': 'wall-other', 'counts': 221}, 
    {'id': 52, 'name': 'ground-other(leaves)', 'counts': 169}, 
    {'id': 50, 'name': 'grass(moss)', 'counts': 154}, 
    {'id': 73, 'name': 'water-other', 'counts': 122}, 
    {'id': 17, 'name': 'hat', 'counts': 104}, 
    {'id': 46, 'name': 'flower', 'counts': 95}, 
    {'id': 59, 'name': 'plant-other', 'counts': 95}, 
    {'id': 42, 'name': 'clouds', 'counts': 90}, 
    {'id': 34, 'name': 'table(dinning_table,desk)', 'counts': 88}, 
    {'id': 31, 'name': 'chair(bench,couch)', 'counts': 74}, 
    {'id': 56, 'name': 'mountain', 'counts': 70}, 
    {'id': 62, 'name': 'rock', 'counts': 66}, 
    {'id': 40, 'name': 'building-other', 'counts': 63}, 
    {'id': 45, 'name': 'floor-other(carpet,mat,rug,platform)', 'counts': 61}, 
    {'id': 25, 'name': 'bowl', 'counts': 47}, 
    {'id': 54, 'name': 'house', 'counts': 45}, 
    {'id': 5, 'name': 'boat', 'counts': 42}, 
    {'id': 53, 'name': 'hill', 'counts': 38}, 
    {'id': 64, 'name': 'sea', 'counts': 37}, 
    {'id': 74, 'name': 'window-other', 'counts': 33}
]

union_ids = [1, 65, 70, 72, 52, 50, 73, 17, 46, 59, 42, 34, 31, 56, 62, 40, 45, 25, 54, 5, 53, 64, 74]
union_names = ['person', 'sky-other', 'tree', 'wall-other', 'ground-other(leaves)', 'grass(moss)',  'water-other',  'hat', 'flower', 'plant-other',  'clouds', 'table(dinning_table, desk)', 'chair(bench,couch)', 'mountain', 'rock', 'building-other', 'floor-other(carpet,mat,rug,platform)', 'bowl', 'house', 'boat', 'hill', 'sea', 'window-other']

def configuration(model, args):
    if args.merged:
        merged = args.merged
        top_k = args.top_k
        background_ignore_index = top_k
        num_classes = top_k
        print("---------------Model Setting---------------")
        print("Model:", model, end="\n\n")
        print("Model Name:", args.model_name, end="\n\n")
        print("1. Top K:", top_k)
        print("2. Number of classes:", top_k)
        print("3. Number of Model output nodes: ", num_classes + 1)
        print("4. Background ignore index:", background_ignore_index)
        print("5. Merged:", merged)
        return model, merged, args.model_name, top_k, num_classes, background_ignore_index
    else:
        top_k = args.top_k
        background_ignore_index = top_k
        num_classes = top_k
        print("---------------Model Setting---------------")
        print("Model:", model, end="\n\n")
        print("Model Name:", args.model_name, end="\n\n")
        print("1. Top K:", top_k)
        print("2. Number of classes:", top_k)
        print("3. Number of Model output nodes: ", num_classes + 1)
        print("4. Background ignore index:", background_ignore_index)
        return model, args.model_name, top_k, num_classes, background_ignore_index

def bgmapper(uniques, labels, bg, main_labels):
    for idx, label in enumerate(uniques):
        if label not in main_labels:
            labels[labels == label] = bg

etri_final = [1, 65, 42, 70, 72, 52, 50, 73]

def label_mapper(labels, top_k, merged = None):
    if merged and merged != "False":
        uniques = np.unique(labels)
        if merged == "etri_merge_top6":
            main_labels = etri_final
            bg = 6
            bgmapper(uniques, labels, bg, main_labels)
            labels[labels == 1] = 0
            labels[labels == 65] = 1
            labels[labels == 42] = 1
            labels[labels == 70] = 2
            labels[labels == 72] = 3
            labels[labels == 52] = 4
            labels[labels == 50] = 4
            labels[labels == 73] = 5
        return labels
    else:
        output_nodes = union_ids[:top_k]
        uniques = np.unique(labels)
        background = top_k
        for idx, unique in enumerate(uniques):
            if unique not in output_nodes:
                labels[labels == unique] = background
            else:
                labels[labels == unique] = output_nodes.index(unique)
        return labels

def get_labelnames(labels):
    uniques = np.unique(labels)
    label_names = []
    for unique in uniques:
        label_names.append(union_names[unique])