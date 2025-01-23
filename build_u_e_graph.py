import json
import random

def build_local_map():
    data_file = '../data/EdNet-1/train_set.json'
    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    # e
    # u
    with open(data_file, encoding='utf8') as i_f:
        data = json.load(i_f)
    u_from_e = '' # e(src) to k(dst)
    e_from_u = '' # k(src) to k(dst)
    print (len(data))

    for line in data:
        exer_id = line['exer_id'] - 1
        user_id = line['user_id'] - 1
        for k in line['knowledge_code']:
            u_from_e += str(exer_id) + '\t' + str(user_id + exer_n) + '\n'
            e_from_u += str(user_id + exer_n) + '\t' + str(exer_id) + '\n'

    '''
    for line in data:
        exer_id = line['exer_id'] - 1  # 练习ID减1，假设从1开始，改为从0开始
        user_id = line['user_id'] - 1  # 用户ID减1，假设从1开始，改为从0开始

        for k in line['knowledge_code']:
            # 使用原始的user_id和exer_id，没有偏移
            u_from_e += str(exer_id) + '\t' + str(user_id) + '\n'
            e_from_u += str(user_id) + '\t' + str(exer_id) + '\n'
'''
    with open('../data/EdNet-1/graph/u_from_e.txt', 'w') as f:
        f.write(u_from_e)
    with open('../data/EdNet-1/graph/e_from_u.txt', 'w') as f:
        f.write(e_from_u)

if __name__ == '__main__':
    build_local_map()
