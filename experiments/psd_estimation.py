import numpy as np
from beamforming import *
from cropac import *

class PSDEParams:
    def __init__(self, sensor_positions, target_direction, k, steering_vectors, covariance_matrices):
        self.sensor_postions = sensor_positions
        self.target_direction = target_direction
        self.k = k
        self.steering_vectors = steering_vectors
        self.covariance_matrices = covariance_matrices

def covariance_psd(w1, Sx, w2):
    return np.maximum(np.real(np.conj(w1[...,None,:]) @ Sx @ w2[...,:,None])[:,0,0], 0.0)

class PSDEstimator:
    def __init__(self, params: PSDEParams):
        self.update_parameters(params)
    def update_parameters(self, params: PSDEParams):
        raise NotImplementedError
    def estimate_psd(self, covariance_matrices):
        raise NotImplementedError
    def update_and_estimate(self, params: PSDEParams, covariance_matrices):
        params.covariance_matrices = covariance_matrices
        self.update_parameters(params)
        return self.estimate_psd(covariance_matrices)

class CH0_PSDE(PSDEstimator):
    def update_parameters(self, params: PSDEParams):
        pass
    def estimate_psd(self, covariance_matrices):
        return np.real(covariance_matrices[:,0,0])

class DAS_PSDE(PSDEstimator):
    def update_parameters(self, params: PSDEParams):
        self.w = wideband_das_weights(params.steering_vectors)
    def estimate_psd(self, covariance_matrices):
        return covariance_psd(self.w, covariance_matrices, self.w)

class MVDR_PSDE(PSDEstimator):
    def __init__(self, params: PSDEParams, LNR):
        self.LNR = LNR
        self.update_parameters(params)
    def update_parameters(self, params: PSDEParams):
        R = diagonal_load(params.covariance_matrices, self.LNR)
        self.w = wideband_mvdr_weights(R, params.steering_vectors)
    def estimate_psd(self, covariance_matrices):
        return covariance_psd(self.w, covariance_matrices, self.w)

class CroPaC_PSDE(PSDEstimator):
    def __init__(self, params: PSDEParams):
        self.update_parameters(params)
    def update_parameters(self, params: PSDEParams):
        self.w1 = blended_ch_das_weights(params.target_direction, 1, params.sensor_postions, params.k)
        self.w2 = blended_ch_das_weights(params.target_direction, 2, params.sensor_postions, params.k)
    def estimate_psd(self, covariance_matrices):
        return covariance_psd(self.w1, covariance_matrices, self.w2)

class DOBEC_PSDE(PSDEstimator):
    def __init__(self, params: PSDEParams, threshold):
        self.threshold = threshold
        self.update_parameters(params)
    def update_parameters(self, params: PSDEParams):
        Rd = diffuse_covariance_matrices(params.sensor_postions, params.k)
        self.w1 = blended_ch_das_weights(params.target_direction, 1, params.sensor_postions, params.k)
        self.w2 = optimal_secondary_weights(self.w1, params.steering_vectors, [Rd])
        self.w_das = wideband_das_weights(params.steering_vectors)
    def estimate_psd(self, covariance_matrices):
        return np.maximum(covariance_psd(self.w1, covariance_matrices, self.w2),
            self.threshold * covariance_psd(self.w_das, covariance_matrices, self.w_das))

class NOBEC_PSDE(PSDEstimator):
    def __init__(self, params: PSDEParams, LNR, threshold):
        self.LNR = LNR
        self.threshold = threshold
        self.update_parameters(params)
    def update_parameters(self, params: PSDEParams):
        R = diagonal_load(params.covariance_matrices, self.LNR)
        self.w1 = blended_ch_das_weights(params.target_direction, 1, params.sensor_postions, params.k)
        self.w2 = optimal_secondary_weights(self.w1, params.steering_vectors, [R])
        self.w_mvdr = wideband_mvdr_weights(R, params.steering_vectors)
    def estimate_psd(self, covariance_matrices):
        return np.maximum(covariance_psd(self.w1, covariance_matrices, self.w2),
            self.threshold * covariance_psd(self.w_mvdr, covariance_matrices, self.w_mvdr))

class DMVDR_PSDE(PSDEstimator):
    def __init__(self, params: PSDEParams, LNR):
        self.LNR = LNR
        self.update_parameters(params)
    def update_parameters(self, params: PSDEParams):
        R = diagonal_load(params.covariance_matrices, self.LNR)
        self.w1 = wideband_mvdr_weights(R, params.steering_vectors)
        self.w2 = optimal_secondary_weights(self.w1, params.steering_vectors, [R], R)
    def estimate_psd(self, covariance_matrices):
        return covariance_psd(self.w1, covariance_matrices, self.w2)

class CroPaC_Gain(PSDEstimator):
    def __init__(self, params: PSDEParams):
        self.update_parameters(params)
    def update_parameters(self, params: PSDEParams):
        self.w1 = blended_ch_das_weights(params.target_direction, 1, params.sensor_postions, params.k)
        self.w2 = blended_ch_das_weights(params.target_direction, 2, params.sensor_postions, params.k)
    def estimate_psd(self, covariance_matrices):
        return 2 * covariance_psd(self.w1, covariance_matrices, self.w2) / (
            covariance_psd(self.w1, covariance_matrices, self.w1) +
            covariance_psd(self.w2, covariance_matrices, self.w2)
        )
