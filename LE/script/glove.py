import numpy as np
import json
import os
import re
import fnmatch

from gensim.scripts.glove2word2vec import glove2word2vec
word2vec_output_file = 'glove.6B.300d.word2vec.txt'
glove_input_file = 'glove.6B.300d.txt'

glove2word2vec(glove_input_file, word2vec_output_file)
print('convert success')

from gensim.models import KeyedVectors

glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

print('model load success')

def take_num(x):
    r=int(re.findall("\d+",x)[0])
    return r

def get_json_file_list(data_dir):
    files = []
    for root, _, file_names in os.walk(data_dir):
        for filename in fnmatch.filter(file_names, '*.json'):
            files.append(os.path.join(root, filename))
    files.sort(reverse=True, key=take_num)
    return files

def glove(text):
    r_list=[]
    r=glove_model.most_similar(text.lower(), topn=3)
    for item in r:
        r_list.append(item[0])
    return r_list

data_dir='../LE/train'
save_dir='../LE/train_pro/'

for dir in get_json_file_list(data_dir):
    file_name=dir[-14:]
    f = json.loads(open(dir, 'r').read())
    options=f['options']
    answers=f['answers']

    for i in range(len(options)):
        o=options[i]
        a=answers[i]
        a_index=ord(a)-ord('A')
        correct=o[a_index].lower()
        if correct in glove_model:
            new=glove(correct)
            new.insert(a_index, correct)
            options[i]=new

    j_data=json.dumps(f, indent=4)

    file_obj=open(save_dir+file_name, 'w')
    file_obj.write(j_data)
    file_obj.close()
    print(file_name+' convert successfully')