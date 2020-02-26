import time

from sklearn.utils import shuffle
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class Iterater:
    #def __init__(self):
    
    
    def train(self, model, optimizer, loss_func, aspect_loss_func, input_tensor, target, target_aspect_u, target_aspect_i):

        optimizer.zero_grad()

        #score = model(batch_path, batch_relation, batch_type, path_num)
        prob, aspect_u, aspect_i = model(input_tensor[0], input_tensor[1])
        #print(score)


        # 損失を計算
        # aspectの二乗誤差と、リンク予測の分類誤差 
        loss = loss_func(prob, target)
        aspect_loss = aspect_loss_func(aspect_u, target_aspect_u)
        aspect_loss += aspect_loss_func(aspect_i, target_aspect_i)
        total_loss = loss + aspect_loss

        # 勾配を計算
        total_loss.backward()

        # 勾配降下
        optimizer.step()


        return float(loss)
    
    

    def iterate_train(self, model, train_data, target_train, user_aspect_dict, item_aspect_dict, n_iter=30, batch_size=2, learning_rate=0.001, print_every=60, plot_every=30):

        print_loss_total = 0
        plot_loss_total = 0
        plot_loss_list = []

        start_time = time.time()

        # 損失関数定義
        loss_func = nn.BCELoss()
        aspect_loss = nn.MSELoss()

        # optimizer 定義
        #optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.002)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # ここから学習
        loss = 0
        for i in range(1, n_iter+1):
            # batchをつくる
            user_id, item_id, target_aspect_u, target_aspect_i, target_batch = self.get_batch(train_data, target_train, batch_size, user_aspect_dict, item_aspect_dict)

            #if len(batch_path_tensor) == 0: continue #pathが一つも取得されなかった場合

            # train
            input_tensor = [user_id, item_id]
            loss = self.train(model, optimizer, loss_func, aspect_loss, input_tensor, target_batch, target_aspect_u, target_aspect_i)
            print_loss_total += loss
            plot_loss_total += loss

            # print_everyごとに現在の平均のlossと、時間、dataset全体に対する進捗(%)を出力
            if i % print_every == 0:
                runtime = time.time() - start_time
                mi, sec = self.time_since(runtime)
                avg_loss = print_loss_total / print_every
                data_percent = int(i * batch_size / train_data.shape[0] * 100)
                print('train loss: {:e}    processed: {}({}%)    {}m{}sec'.format(avg_loss, i*batch_size, data_percent, mi, sec))
                print_loss_total = 0

            # plot_everyごとplot用のlossをリストに記録しておく
            if i % plot_every == 0:
                avg_loss = plot_loss_total / plot_every
                plot_loss_list.append(avg_loss)
                plot_loss_total = 0


        return plot_loss_list
        # return print_loss_total
        
        
    def get_batch(self, data, target, batch_size, user_aspect_dict, item_aspect_dict):

        idx = np.random.randint(0, len(data), batch_size) #重複をゆるしている
        user_id = []
        item_id = []
        user_aspect = []
        item_aspect = []

        for path in data[idx]:
            user_id.append(path[1])
            item_id.append(path[0])
            user_aspect.append(user_aspect_dict[path[1]])
            item_aspect.append(item_aspect_dict[path[0]])

        user_id = torch.tensor(user_id, dtype=torch.long, device=device)
        item_id = torch.tensor(item_id, dtype=torch.long, device=device)
        user_aspect = torch.tensor(user_aspect, device=device)
        item_aspect = torch.tensor(item_aspect, device=device)

        batch_target = torch.tensor(target[idx], dtype=torch.float, device=device)

        return user_id, item_id, user_aspect, item_aspect, batch_target

    
    def time_since(self, runtime):
        mi = int(runtime / 60)
        sec = int(runtime - mi * 60)
        return (mi, sec)