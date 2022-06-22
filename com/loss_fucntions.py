import numpy as np
def cross_entropy_error(predict_output_vector,raw_output_vector):
    h = 1e-4
    return -np.sum(raw_output_vector*np.log(predict_output_vector + h))

def mean_square_erorr(raw_output_vector,predict_output_vector):
    return 