import numpy as np
# df = pd.read_csv('/home/nabeel/Downloads/multilabelMetrics/MLTSVM.csv')

# actual = df['actual']
# pred   = df['predicted']

# yi_actual = [[1,1,1,0],[1,1,1,0],[1,0,1,1]]
# xi_pred   = [[1,1,1,0],[1,0,0],[1,1,1,0]]



# y_test , predictions = [] , []
# for i,j in zip(actual,pred):
#     y_test.append(i[1:-1].split(" "))
#     predictions.append(j[1:-1].split(" "))

#================ ACCURACY =========================
# a , p = [] , []
# for i,j in zip(actual,pred):
#     a.append(i[1:-1])
#     p.append(j[1:-1])
#
#
# accuracy  = 0
#
# correct = 0
#
# for i,j in zip(a,p):
#     if i==j:
#         correct+=1
#
#
# print("total = ",len(y_test))
# print("TP+TN = ",correct)
#
# accuracy = correct/len(y_test)
# print("accuracy = ",accuracy*100)
#==============================================
pr = [[0.3321,0.9677,0.5211,0.5,0.4000],[0.321,0.3677,0.5211,0.5,0.4000]]
ac = [[3,2],[2,0,1]]
#=========================START SECTION predicted scores to one hot encoding ============================================


def pred_scores_converter(a):
    a = list(a)
    xi_pred = []
    total_labels=len(a[0])
    temp_list =total_labels*[0]
    count = 0
    for k in a:
        k = list(k)
        for i in k:
            if i > 0.5:
                # print(k.index(i))
                temp_list.insert(k.index(i), 1)
            if len(temp_list) > total_labels:
                temp_list.pop()
        xi_pred.append(temp_list)
        temp_list = total_labels*[0]
    return xi_pred

#=========================END SECTION predicted scores to one hot encoding ============================================

#=========================START SECTION actual labels to one hot encoding ============================================


def true_label_convert(a,b):
    b = list(b)
    yi_true = []
    total_labels=len(a[0])
    temp_list = total_labels*[0]
    count = 0
    for k in b:
        k = list(k)
        k = list(np.sort(k))
        for i in k:
            if k != '':
                # print(i)
                temp_list.insert(i,1)
            if len(temp_list) > total_labels:
                temp_list.pop()
        yi_true.append(temp_list)
        temp_list = total_labels*[0]
    return yi_true

#=========================END SECTION actual labels to one hot encoding ============================================
def take_values(predicted,actual):
    a, b = predicted,actual
    return pred_scores_converter(a),true_label_convert(a,b)

# a,b = take_values(pr,ac)
#
# y_test = np.array(true_label_convert(b))
# predictions = np.array(pred_scores_converter(a))
# #
# #
# #
# #     # print(y_test.shape)
# # # print(predictions.shape)
# # #
# print("Accuracy         = {:.2f} %\t".format(accuracy(y_test,predictions)*100))
# print("Precision        = {:.2f} %\t".format(precision(y_test,predictions)*100))
# print("Recall           = {:.2f} %\t".format(recall(y_test,predictions)*100))
# print("Subset Accuracy  = {:.2f} %\t".format(subsetAccuracy(y_test,predictions)*100))
# print("Hamming Loss     = {:.2f} %\t".format(hammingLoss(y_test,predictions)*100))
# print("f-beta           = {:.2f} %\t".format(fbeta(y_test,predictions)))
# print("f1_score         = {:.2f} %\t".format(f1_score(y_test,predictions)))