# KoopmanLab learning and implementation plan

Do not wait until you have studied every topic. Learn the material in the order
below and immediately use each idea in the project.

## Phase 1: simulation foundations

Learn:

- Python functions, modules, arrays, slicing, and plotting
- State variables and first-order ODE systems
- Initial conditions and parameters
- Euler's method conceptually
- What `solve_ivp` computes
- Time-series plots, phase portraits, and mechanical energy

Build:

- Run and understand the supplied pendulum simulation
- Change parameters and predict the result before running it
- Explain why damping removes energy
- Add one new plot without copying a completed solution

Completion check:

- You can explain every quantity returned by `simulate_pendulum`.
- You understand why the trajectory array has shape `(number of times, 2)`.
- You can distinguish the physical model from the numerical solver.

## Phase 2: data and linear algebra

Learn:

- Vectors, matrices, matrix multiplication, norms, and transpose
- Least squares and the pseudoinverse
- Eigenvalues, eigenvectors, and discrete-time stability
- Singular value decomposition at a conceptual level
- How to organize trajectory data without leakage

Build:

- Generate hundreds of trajectories from randomized initial conditions
- Split entire trajectories into training, validation, and test sets
- Construct snapshot matrices `X = [x_0, ..., x_(T-1)]` and
  `Y = [x_1, ..., x_T]`
- Fit DMD using `A = Y @ pinv(X)`
- Compare one-step and recursive rollout predictions

Completion check:

- You can explain why DMD is a least-squares linear model.
- You can interpret eigenvalue magnitudes of the learned matrix.
- You understand why low one-step error may coexist with poor rollouts.

## Phase 3: introductory machine learning

Learn:

- Inputs, targets, parameters, predictions, and loss
- Training, validation, and test sets
- Dense neural layers and activation functions
- Gradient descent and backpropagation conceptually
- Epochs, batches, learning rate, and overfitting
- PyTorch tensors, modules, optimizers, and checkpoints

Build:

- Normalize states using training-set statistics only
- Train an MLP to predict `x_(t+1)` from `x_t`
- Plot training and validation loss
- Evaluate recursive rollouts on untouched test trajectories

Completion check:

- You can write a basic PyTorch training loop.
- You can explain what every tensor dimension represents.
- You can recognize overfitting from loss curves.

## Phase 4: Koopman learning

Learn:

- Observables and the Koopman operator
- Why the exact operator is linear but generally infinite-dimensional
- Autoencoders and latent representations
- Reconstruction, latent-linearity, prediction, and multistep losses
- Eigenanalysis of the learned latent matrix

Build:

- Encoder `z = E(x)`
- Linear latent step `z_next = K z`
- Decoder `x_hat = D(z)`
- Combined training objective
- Multistep latent rollout

Completion check:

- You can explain what the encoder, matrix, and decoder each contribute.
- You know that a learned latent variable is not automatically physically
  meaningful or a true Koopman eigenfunction.
- You compare Koopman learning fairly against DMD and an MLP.

## Phase 5: research-quality evaluation

Run controlled experiments on:

- Rollout horizon
- Latent dimension
- Training-set size
- Measurement noise
- Initial conditions outside the training region
- Multiple random seeds

Report:

- One-step MSE
- Error versus rollout horizon
- Stable prediction horizon
- Phase-portrait agreement
- Energy behavior
- Parameter count and inference time
- Mean and standard deviation across seeds
- Representative failure cases

## Suggested weekly rhythm

- 2 hours: learn one small theory topic
- 4 hours: implement it
- 2 hours: test and debug
- 1 hour: write notes and update the README
- 1 hour: review results and decide the next experiment

At the end of every session, write three sentences: what you changed, what you
observed, and what remains uncertain.

