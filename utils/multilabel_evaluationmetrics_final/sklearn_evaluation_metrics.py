from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, hamming_loss


def accuracy(y_test, predictions):
    accuracy=accuracy_score(y_test, predictions)
    return accuracy


def precision(y_test, predictions):
    """
    Precision of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    precision : float
        Precision of our model
    """


    return precision_score(y_test, predictions, average="weighted")


def recall(y_test, predictions):

    return recall_score(y_test, predictions, average="weighted")



def f1_scor(y_test, predictions):

    return f1_score(y_test, predictions, average="weighted")

def subsetAccuracy(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    return accuracy
def hammingLoss(y_test, predictions):
    return hamming_loss(y_test, predictions)