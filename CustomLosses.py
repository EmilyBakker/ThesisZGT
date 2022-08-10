import tensorflow.python.keras.backend as K
from tensorflow.python.keras.losses import binary_crossentropy
#from utils.global_objectives.loss_layers_v2 import precision_recall_auc_loss
import tensorflow as tf



def target_replication_loss(alpha: float, mask_value=-1., class_weights: dict=None):
    """
    Custom loss function to implement target replication.
    Args:
        alpha (float): Value between 0 and 1, denoting the relative importance of earlier predictions
    """     

    def inner_loss(y_true, y_pred):
        mask = K.equal(y_true, mask_value)
        mask = 1 - K.cast(mask, K.floatx())

        losses = K.binary_crossentropy(y_true, y_pred) * mask
        
        pre_final_avg = K.sum(losses[:,:-1], axis=-1) / (K.sum(mask[:, :-1], axis=-1))
        final_loss = losses[:, -1]

        tr_loss = (alpha*pre_final_avg) + ((1-alpha)*final_loss)
        tr_loss = K.expand_dims(tr_loss)

        
        # Importance to longer sequences
        #tr_loss = (tr_loss * K.sum(mask, keepdims=True, axis=-1)) / maxlen 

        # Importance to shorter sequences
        #tr_loss = (tr_loss * maxlen) / K.sum(mask, keepdims=True, axis=-1) 

        if class_weights is not None:
            b = class_weights.get(0)
            a = class_weights.get(1) - b
            weigths = (a*y_true[:,-1]) + b
            weigths = K.expand_dims(weigths)
            tr_loss = weigths*tr_loss

        return tr_loss
    return inner_loss


def custom_binary_entropy_loss(class_weights=None, label_smoothing=0):
    def inner_loss(y_true, y_pred):
        loss = binary_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
        loss = K.expand_dims(loss)

        if class_weights is not None:
            b = class_weights.get(0)
            a = class_weights.get(1) - b
            weigths = (a*y_true[:,-1]) + b
            weigths = K.expand_dims(weigths)
            loss = weigths*loss

        return loss   
    return inner_loss



#from utils.global_objectives import util_v2 as util
def pr_loss(class_weights=None):
    def inner_loss(y_true, y_pred):
        get_num_labels = lambda labels : 1
        loss = precision_recall_auc_loss(y_true, y_pred)[0]

        if class_weights is not None:
            b = class_weights.get(0)
            a = class_weights.get(1) - b
            weigths = (a*y_true[:,-1]) + b
            weigths = K.expand_dims(weigths)
            loss = weigths*loss

        return loss   
    return inner_loss
