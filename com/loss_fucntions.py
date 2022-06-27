import numpy as np
def cross_entropy_error(predict_output_vector,raw_output_vector):
    h = 1e-4
    return -np.sum(raw_output_vector*np.log(predict_output_vector + h))

def mean_square_erorr(predict_output_vector,raw_output_vector):
    return 0.5 * np.sum((predict_output_vector - raw_output_vector)**2)

def mean_square_error_batch(predict_output_vector,raw_output_vector):
    size_of_batch = predict_output_vector.shape[0]
    return (1/size_of_batch)*(0.5 * np.sum((predict_output_vector - raw_output_vector)**2))
