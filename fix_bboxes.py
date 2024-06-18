# One version of the tool wrote bounding box annotations as absolute values instead of relative
# this fixes that

import os

labels_path = "D:\\Interns\\Prabhanjan\\keypoint-annotation-tool\\trial2\\labels\\"
for i in os.listdir(labels_path):
    with open(labels_path + i) as f:
        text = f.readlines()[0].split(' ')
        for j in range(1, 5):
            text[j] = str(float(text[j]) / 256)
        for j in range(len(text)):
            if float(text[j]) < 0:
                text[j] = str(0)
            elif float(text[j]) > 1:
                text[j] = str(1)
        text = ' '.join(text)
        print(text)
    with open(labels_path + i, "w") as f:
        f.write(text)