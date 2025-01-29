objects_mapping = {
    "person":{"left": 1, "right": 1},
    "cat": {"left": 1, "right": 2},
    "dog": {"left": 1, "right": 3},
    "horse": {"left": 1, "right": 4},
    "elephant": {"left": 1, "right": 5},
    "zebra": {"left": 1, "right": 12},
    "giraffe": {"left": 1, "right": 13},
    "bird": {"left": 1, "right": 14},
    "bicycle": {"left": 2, "right": 1},
    "car": {"left": 2, "right": 2},
    "motorcycle": {"left": 2, "right": 3},
    "stop sign": {"left": 2, "right": 4},
    "bus": {"left": 2, "right": 5},
    "airplane": {"left": 2, "right": 12},
    "train": {"left": 2, "right": 13},
    "truck": {"left": 2, "right": 14},
    "boat": {"left": 2, "right": 15},
    "traffic light": {"left": 2, "right": 23},
    "fire hydrant": {"left": 2, "right": 24},
    "parking meter": {"left": 2, "right": 34},
    "bench": {"left": 2, "right": 35},
    "chair": {"left": 3, "right": 12},
    "couch": {"left": 3, "right": 13},
    "potted plant": {"left": 3, "right": 14},
    "bed": {"left": 3, "right": 23},
    "dining table": {"left": 3, "right": 24},
    "toilet": {"left": 3, "right": 25},
    "tv": {"left": 3, "right": 34},
    "laptop": {"left": 3, "right": 123},
    "mouse": {"left": 3, "right": 124},
    "remote": {"left": 3, "right": 125},
    "keyboard": {"left": 3, "right": 134},
    "cell phone": {"left": 3, "right": 135},
    "microwave": {"left": 3, "right": 234},
    "oven": {"left": 3, "right": 235},
    "toaster": {"left": 3, "right": 245},
    "sink": {"left": 3, "right": 345},
    "refrigerator": {"left": 3, "right": 123},
    "book": {"left": 3, "right": 124},
    "clock": {"left": 3, "right": 125},
    "vase": {"left": 3, "right": 134},
    "scissors": {"left": 3, "right": 135},
    "teddy bear": {"left": 3, "right": 234},
    "sports ball": {"left": 3, "right": 345},
    "frisbee": {"left": 3, "right": 234},
    "skis": {"left": 3, "right": 235},
    "snowboard": {"left": 3, "right": 245},
    "tennis racket": {"left": 4, "right": 145},
    "baseball bat": {"left": 4, "right": 124},
    "baseball glove": {"left": 4, "right": 125},
    "skateboard": {"left": 4, "right": 134},
    "surfboard": {"left": 4, "right": 135},
    "tie": {"left": 4, "right": 134},
    "suitcase": {"left": 4, "right": 135},
    "pizza": {"left": 4, "right": 123},
    "donut": {"left": 4, "right": 124},
    "cake": {"left": 4, "right": 125},
    "banana": {"left": 4, "right": 134},
    "apple": {"left": 4, "right": 135},
    "sandwich": {"left": 4, "right": 145},
    "orange": {"left": 4, "right": 234},
    "broccoli": {"left": 4, "right": 235},
    "carrot": {"left": 4, "right": 245},
    "hot dog": {"left": 4, "right": 345},
    "fork": {"left": 4, "right": 345},
    "knife": {"left": 4, "right": 123},
    "spoon": {"left": 4, "right": 124},
    "bowl": {"left": 4, "right": 125},
    "bottle": {"left": 4, "right": 234},
    "wine glass": {"left": 4, "right": 235},
    "cup": {"left": 4, "right": 245},
    "kite": {"left": 4, "right": 123},
    "tennis racket": {"left": 4, "right": 145}
}


class_labels = {0: { "name":'person', "priority":0 },
                1: { "name" :'bicycle',  "priority":1 },
                2: { "name" :'car',  "priority":1 },
                3: { "name" :'motorcycle',  "priority":2 },
                4: { "name" :'airplane',  "priority":1 },
                5: { "name" :'bus',  "priority":3 },
                6: { "name" :'train',  "priority":1 },
                7: { "name" :'truck',  "priority":1 },
                8: { "name" :'boat',  "priority":1 },
                9: { "name" :'traffic light',  "priority":1 },
                10:{ "name" : 'fire hydrant',  "priority":1 },
                11:{ "name" : 'stop sign',  "priority":1 },
                12:{ "name" : 'parking meter',  "priority":1 },
                13:{ "name" : 'bench',  "priority":1 },
                14:{ "name" : 'bird',  "priority":1 },
                15:{ "name" : 'cat',  "priority":1 },
                16:{ "name" : 'dog',  "priority":1 },
                17:{ "name" : 'horse',  "priority":1 },
                18:{ "name" : 'sheep',  "priority":1 },
                19:{ "name" : 'cow',  "priority":1 },
                20:{ "name" : 'elephant',  "priority":1 },
                21:{ "name" : 'bear',  "priority":1 },
                22:{ "name" : 'zebra',  "priority":1 },
                23:{ "name" : 'giraffe',  "priority":1 },
                24:{ "name" : 'backpack',  "priority":1 },
                25:{ "name" : 'umbrella',  "priority":1 },
                26:{ "name" : 'handbag',  "priority":1 },
                27:{ "name" : 'tie',  "priority":1 },
                28:{ "name" : 'suitcase',  "priority":1 },
                29:{ "name" : 'frisbee',  "priority":1 },
                30:{ "name" : 'skis',  "priority":1 },
                31:{ "name" : 'snowboard',  "priority":1 },
                32:{ "name" : 'sports ball',  "priority":1 },
                33:{ "name" : 'kite',  "priority":1 },
                34:{ "name" : 'baseball bat',  "priority":1 },
                35:{ "name" : 'baseball glove',  "priority":1 },
                36:{ "name" : 'skateboard',  "priority":1 },
                37:{ "name" : 'surfboard',  "priority":1 },
                38:{ "name" : 'tennis racket',  "priority":1 },
                39:{ "name" : 'bottle',  "priority":1 },
                40:{ "name" : 'wine glass',  "priority":1 },
                41:{ "name" : 'cup',  "priority":1 },
                42:{ "name" : 'fork',  "priority":1 },
                43:{ "name" : 'knife',  "priority":1 },
                44:{ "name" : 'spoon',  "priority":1 },
                45:{ "name" : 'bowl',  "priority":1 },
                46:{ "name" : 'banana',  "priority":1 },
                47:{ "name" : 'apple',  "priority":1 },
                48:{ "name" : 'sandwich',  "priority":1 },
                49:{ "name" : 'orange',  "priority":1 },
                50:{ "name" : 'broccoli',  "priority":1 },
                51:{ "name" : 'carrot',  "priority":1 },
                52:{ "name" : 'hot dog',  "priority":1 },
                53:{ "name" : 'pizza',  "priority":1 },
                54:{ "name" : 'donut',  "priority":1 },
                55:{ "name" : 'cake',  "priority":1 },
                56:{ "name" : 'chair',  "priority":1 },
                57:{ "name" : 'couch',  "priority":1 },
                58:{ "name" : 'potted plant',  "priority":1 },
                59:{ "name" : 'bed',  "priority":1 },
                60:{ "name" : 'dining table',  "priority":1 },
                61:{ "name" : 'toilet',  "priority":1 },
                62:{ "name" : 'tv',  "priority":1 },
                63:{ "name" : 'laptop',  "priority":1 },
                64:{ "name" : 'mouse',  "priority":1 },
                65:{ "name" : 'remote',  "priority":1 },
                66:{ "name" : 'keyboard',  "priority":1 },
                67:{ "name" : 'cell phone',  "priority":1 },
                68:{ "name" : 'microwave',  "priority":1 },
                69:{ "name" : 'oven',  "priority":1 },
                70:{ "name" : 'toaster',  "priority":1 },
                71:{ "name" : 'sink',  "priority":1 },
                72:{ "name" : 'refrigerator',  "priority":1 },
                73:{ "name" : 'book',  "priority":1 },
                74:{ "name" : 'clock',  "priority":1 },
                75:{ "name" : 'vase',  "priority":1 },
                76:{ "name" : 'scissors',  "priority":1 },
                77:{ "name" : 'teddy bear', "priority":1 },
                78:{ "name" : 'hair drier',  "priority":1 },
                79:{ "name" : 'toothbrush', "priority":1 },
                }
