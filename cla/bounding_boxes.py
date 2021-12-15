import json
import pickle


def get_index(number):
    if number == 0:
        return '00000'
    digits = 0
    res = ''
    cp = number
    while cp > 0:
        digits += 1
        cp = (int)(cp / 10)

    while digits < 5:
        res += '0'
        digits += 1

    res += str(number)

    return res

n = 8500
json_file_path = lambda idx : f'../dataset/jsons/rs19_val/rs{idx}.json'
bbs = {}
for i in range(n):
    bbs[i] = []
    file = open(json_file_path(get_index(i)))
    data = json.load(file)
    for obj in data['objects']:
        if 'boundingbox' in obj:
            lst = list(obj['boundingbox'])
            if obj['label'] == "switch-left":
                lst.extend([0])
                bbs[i].append(lst)
            elif obj['label'] == "switch-right":
                lst.extend([1])
                bbs[i].append(lst)

for key in bbs:
    print(key, '->', bbs[key])

with open('bounding_boxes_dict.pickle', 'wb') as handle:
    pickle.dump(bbs, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('bounding_boxes_dict.pickle', 'rb') as handle:
#     b = pickle.load(handle)