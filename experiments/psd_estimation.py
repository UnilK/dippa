import numpy as np
from beamforming import *
from cropac import *
from util import *

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
    def __init__(self, params: PSDEParams = None):
        if params is not None:
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
    def __init__(self, LNR, params: PSDEParams = None):
        self.LNR = LNR
        super().__init__(params)
    def update_parameters(self, params: PSDEParams):
        R = diagonal_load(params.covariance_matrices, self.LNR)
        self.w = wideband_mvdr_weights(R, params.steering_vectors)
    def estimate_psd(self, covariance_matrices):
        return covariance_psd(self.w, covariance_matrices, self.w)

class CroPaC_PSDE(PSDEstimator):
    def update_parameters(self, params: PSDEParams):
        self.w1 = blended_ch_das_weights(params.target_direction, 1, params.sensor_postions, params.k)
        self.w2 = blended_ch_das_weights(params.target_direction, 2, params.sensor_postions, params.k)
    def estimate_psd(self, covariance_matrices):
        return covariance_psd(self.w1, covariance_matrices, self.w2)

class DOBEC_PSDE(PSDEstimator):
    def __init__(self, threshold, params: PSDEParams = None, order=1):
        self.threshold = threshold
        self.order = order
        super().__init__(params)
    def update_parameters(self, params: PSDEParams):
        Rd = diffuse_covariance_matrices(params.sensor_postions, params.k)
        self.w1 = blended_ch_das_weights(params.target_direction, self.order, params.sensor_postions, params.k)
        self.w2 = optimal_secondary_weights(self.w1, params.steering_vectors, [Rd])
        self.w_das = wideband_das_weights(params.steering_vectors)
    def estimate_psd(self, covariance_matrices):
        return np.maximum(covariance_psd(self.w1, covariance_matrices, self.w2),
            self.threshold * covariance_psd(self.w_das, covariance_matrices, self.w_das))

class NOBEC_PSDE(PSDEstimator):
    def __init__(self, LNR, threshold, params: PSDEParams = None, order=1, mvdrLNR=None):
        self.LNR = LNR
        self.mvdrLNR = mvdrLNR
        if mvdrLNR is None:
            self.mvdrLNR = self.LNR
        self.threshold = threshold
        self.order = order
        super().__init__(params)
    def update_parameters(self, params: PSDEParams):
        R = diagonal_load(params.covariance_matrices, self.LNR)
        self.w1 = blended_ch_das_weights(params.target_direction, self.order, params.sensor_postions, params.k)
        self.w2 = optimal_secondary_weights(self.w1, params.steering_vectors, [R])
        if self.threshold > 0.0:
            R = diagonal_load(params.covariance_matrices, self.mvdrLNR)
            self.w_mvdr = wideband_mvdr_weights(R, params.steering_vectors)
    def estimate_psd(self, covariance_matrices):
        psd = covariance_psd(self.w1, covariance_matrices, self.w2)
        if self.threshold > 0.0:
            psd = np.maximum(psd, self.threshold * covariance_psd(self.w_mvdr, covariance_matrices, self.w_mvdr))
        return psd

class DMVDR_PSDE(PSDEstimator):
    def __init__(self, LNR, params: PSDEParams = None):
        self.LNR = LNR
        super().__init__(params)
    def update_parameters(self, params: PSDEParams):
        R = diagonal_load(params.covariance_matrices, self.LNR)
        self.w1 = wideband_mvdr_weights(R, params.steering_vectors)
        self.w2 = optimal_secondary_weights(self.w1, params.steering_vectors, [R], R)
    def estimate_psd(self, covariance_matrices):
        return covariance_psd(self.w1, covariance_matrices, self.w2)

class CroPaC_Gain(PSDEstimator):
    def __init__(self, threshold, params: PSDEParams = None):
        self.threshold = threshold
        super().__init__(params)
    def update_parameters(self, params: PSDEParams):
        self.w1 = blended_ch_das_weights(params.target_direction, 1, params.sensor_postions, params.k)
        self.w2 = blended_ch_das_weights(params.target_direction, 2, params.sensor_postions, params.k)
    def estimate_psd(self, covariance_matrices):
        return np.maximum(self.threshold, 2 * covariance_psd(self.w1, covariance_matrices, self.w2) / (
            covariance_psd(self.w1, covariance_matrices, self.w1) +
            covariance_psd(self.w2, covariance_matrices, self.w2)
        ))

class CroPaC_PSDE(PSDEstimator):
    def __init__(self, threshold, params: PSDEParams = None):
        self.threshold = threshold
        super().__init__(params)
    def update_parameters(self, params: PSDEParams):
        self.w1 = blended_ch_das_weights(params.target_direction, 1, params.sensor_postions, params.k)
        self.w2 = blended_ch_das_weights(params.target_direction, 2, params.sensor_postions, params.k)
        self.w_das = wideband_das_weights(params.steering_vectors)
    def estimate_psd(self, covariance_matrices):
        gain = np.maximum(self.threshold, 2 * covariance_psd(self.w1, covariance_matrices, self.w2) / (
            covariance_psd(self.w1, covariance_matrices, self.w1) +
            covariance_psd(self.w2, covariance_matrices, self.w2)
        ))
        return covariance_psd(self.w_das, covariance_matrices, self.w_das) * gain**2
        


def si_psd_sdr(noise_snaps, target_snaps, estimate_snaps):

    sq_n = np.sqrt(noise_snaps)
    sq_t = np.sqrt(target_snaps)
    sq_e = np.sqrt(estimate_snaps)

    beta = np.sum(sq_e * sq_n) / np.sum(sq_n * sq_n)
    sq_et = sq_e - sq_n * beta

    alpha = np.sum(sq_et * sq_t) / np.sum(sq_et * sq_et + sq_n * sq_n)

    return pow2db(np.sum(sq_t**2)) - pow2db(np.sum((sq_t - alpha * sq_et)**2) + alpha**2 * np.sum(sq_n**2))
    
    #return pow2db(np.sum(np.maximum(0, sq_et**2 * sq_t**2))) - pow2db(np.sum(sq_n**2))
    # return pow2db(np.sum(sq_et*sq_t)) - pow2db(np.sum(sq_n**2))
    # return pow2db(np.sum(sq_et**2)) - pow2db(np.sum(sq_n**2))
    # return pow2db(np.sum(sq_t**2)) - pow2db(np.sum((sq_et - sq_t)**2)) - pow2db(np.sum(sq_n**2))

    # scale-invariant signal-to-distortion ratio for PSD estimates
    # return pow2db(np.sum((alpha * sq_t)**2) / np.sum((alpha * sq_t - sq_e)**2))