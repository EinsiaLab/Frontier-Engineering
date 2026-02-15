import numpy as np
from code_linear import *
from ORBGRAND import ORBGRANDDecoder
from SGRAND import SGRANDDecoder, GRANDDecoder
from chase import ChaseDecoder

def triple_decoder_analysis_step(code, decoder, lb=np.array([0,0,0]), ub=np.array([3,3,3]), num_samples=1e4):
    """
    Analyze the triple decoders.
    The closest error can be represented as a triplet (n1, n2, n3).
    Find the closest error that has the least norm.
    """
    neighbors = code.get_nearest_neighbors()
    neighbor = neighbors[0]
    # find the 3 positions that neighbor is not 0
    positions = np.where(neighbor != 0)[0]

    noises = lb + (ub - lb) * np.random.rand(int(num_samples), 3)
    noises = np.sort(noises, axis=1)
    rx_signals = -1 * np.ones((int(num_samples), code.n))
    for i in range(3):
        # scale the noise to the distance of the code
        rx_signals[:, positions[i]] += noises[:, i]

    # decode the received signals
    decoded = code.decode(rx_signals)
    correct = np.all(decoded == 0, axis=1)
    if np.sum(~correct) == 0:
        print("all results are correct!")
        return 0, None, None
    wrong_noises = noises[~correct]
    dists = np.linalg.norm(wrong_noises, axis=1)
    # find the closest error
    min_index = np.argmin(dists)
    closest_error = wrong_noises[min_index]
    min_dist = dists[min_index]

    # among the correct results, find those are strictly smaller than the closest error
    smaller = np.all(noises < closest_error, axis=1)
    smaller_noises = noises[smaller & correct]
    if smaller_noises.shape[0] == 0:
        new_lb = lb
    else:
        max_index = np.argmax(np.linalg.norm(smaller_noises, axis=1))
        new_lb = smaller_noises[max_index]
    return min_dist, closest_error, new_lb

def triple_decoder_analysis(code, decoder, iters=5, num_samples=1e4):
    code.set_decoder(decoder)
    lb = np.array([0, 0, 0])
    ub = np.array([3, 3, 3])
    for i in range(iters):
        min_dist, ub, lb = triple_decoder_analysis_step(code, decoder, 0.9*lb, 1.1*ub, num_samples)
        if ub is None:
            print("No valid ub found.")
            break
        print(ub, min_dist)
    return min_dist, ub

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    # Example usage
    r = 4
    code = HammingCode(r)
    # decoder = SGRANDDecoder(code)
    # decoder = ORBGRANDDecoder(code)
    # decoder = GRANDDecoder(code)
    decoder = ChaseDecoder(code=code, t=2)
    min_dist, ub = triple_decoder_analysis(code, decoder, iters=10, num_samples=1e4)