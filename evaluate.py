from sklearn.utils import shuffle
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

class Evalueter:
    #def __init__(self, test_data, user_size, metric):
    def __init__(self, user_size, metric):
        self.user_size = user_size
        self.metric = metric
        #self.test_data = test_data
        
    # Cythonでやりたい
    def get_user_rankinglist(data, target_user):
        ranking_list = []
        for d in data:
            if target_user == d[0] or target_user == d[-1]:
                ranking_list.append(d)

        return ranking_list
        
        
    def evaluate_ranking(model, test_posi, test_nega):
        with torch.no_grad():
            #以下をforで回す
            ranking_score_list = []
            ranking_score_list1 = []
            ranking_score_list2 = []
            count = 0

            for idx in range(user_size):

                # あるuserのposiなランキングリストをテストデータから持ってくる
                # nega なランキングリストも持ってくる
                posi_ranking_list = get_user_rankinglist(test_posi, idx)
                nega_ranking_list = get_user_rankinglist(test_nega, idx)

                # ranking_listを取得できなかった場合
                if len(posi_ranking_list) == 0 or len(nega_ranking_list) == 0:
                    continue

                # ランキングリストをバッチ化
                batch_data = np.array(posi_ranking_list + nega_ranking_list)
                target_batch = [1 for i in range(len(posi_ranking_list))] + [0 for i in range(len(nega_ranking_list))]


                # modelに入力できる形にする
                user_id = torch.tensor([usr for usr in batch_data[:, 0]], dtype=torch.long, device=device)
                item_id = torch.tensor([item for item in batch_data[:, 1]], dtype=torch.long, device=device)

                # ランキングリストをmodelに入力
                prob, _, _ = model(user_id, item_id)
                score_list = np.array([prob[i].item() for i in range(len(prob))])


                # PR-AUC, ROC-AUC, NDCGを計算する
                if self.metric == 'map': # sklearn.metrics.average_precision_score
                    ap = average_precision_score(target_batch, score_list)
                    ranking_score_list1.append(ap) 

                    roc = roc_auc_score(target_batch, score_list)
                    ranking_score_list2.append(roc) 


                #count += 1
                #if count > stop_count:
                #    break

                #if count > 30:
                #    break

                #count += 1

        return np.mean(np.array(ranking_score_list1)), np.mean(np.array(ranking_score_list2))
    