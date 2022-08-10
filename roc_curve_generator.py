def save_roc_curve(experiment_name ='', y_true = None, y_predict = None):
    """
    funtion for generating and saving roc curve, 

    Parameters
    ----------
    experiment_name : TYPE, optional
        DESCRIPTION. The default is ''.
    y_true : TYPE, optional
        DESCRIPTION. The default is None.
    y_predict : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve
    
    assert (len(y_true) == len(y_predict))
    
    fig, ax = plt.subplots(figsize=(12,8))
    
    fpr,tpr,thr = roc_curve(y_true, y_predict, pos_label=1)
    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    
    
    ax.plot(fpr, tpr, color='b',
            lw=2, alpha=.8)
    
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic")
    ax.legend(loc="lower right")
    ax.set_ylabel('Sensitivity')
    ax.set_xlabel('1-Specificity')
    
    # check and create roc curves folder
    if not os.path.isdir('/mnt/data/embakker/results/roc_curves'):
        os.makedirs('/mnt/data/embakker/results/roc_curves')
    
    fname_roc = '/mnt/data/embakker/results/roc_curves/roc_curve_'+experiment_name+'.png'
    
    # avoid overwriting
    # dummy_version_nr = 0
    # while(os.path.isfile(fname_roc)):
    #     dummy_version_nr += 1
    #     fname_roc = 'roc_curves/roc_curve_'+experiment_name+'('+str(dummy_version_nr)+').png' 
        
        
    # save figure
    plt.savefig(fname_roc)
    
def main():
    import numpy as np
    y_true = np.array([1]*100 + [0]*100)
    y_predict = np.array(np.random.rand(200))
    save_roc_curve(experiment_name = 'testing_save_roc_curve',
                     y_true = y_true,
                     y_predict = y_predict)
    

if __name__ == '__main__':
    main()
