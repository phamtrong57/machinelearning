import numpy as np
def cross_entropy_error(raw_output_vector,predict_output_vector):
    return -np.sum(raw_output_vector*np.log(predict_output_vector))

def mean_square_erorr(raw_output_vector,predict_output_vector):
    return 