import numpy as np

def forward_algorithm(A:np.ndarray, B:np.ndarray, pi:np.ndarray, observations:list):
    steps = len(observations)
    num_of_states, _ = B.shape
    Alpha = np.zeros((steps, num_of_states))
    for step in range(steps):
        for i in range(num_of_states):
            if step == 0:
                Alpha[step][i] = pi[i] * B[i][observations[step]]
            else:
                for j in range(num_of_states):
                    Alpha[step][i] += Alpha[step-1][j] * A[j][i]
                Alpha[step][i] *= B[i][observations[step]]
    posibility = 0.
    for i in range(num_of_states):
        posibility += Alpha[-1][i]
    return posibility

if __name__ == "__main__":
    A = np.array([[0.5, 0.25, 0.25],
                  [0.375, 0.25, 0.375],
                  [0.25, 0.125, 0.625]])
    B = np.array([[0.8, 0.2],
                  [0.5, 0.5],
                  [0.1, 0.9]])
    pi = np.array([0.2, 0.4, 0.4])
    observations = [0, 1, 1]
    print("posibility: ", forward_algorithm(A, B, pi, observations))