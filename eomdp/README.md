# Policy Computation and Simulation

This library includes functions to compute weak classifier entropies, map them
to offloading metric values, compute optimal policy thresholds under the MDP
framework, and to simulate these policies under a token bucket.

# utils.py

- `cost(wrank, srank, ctype)`: For N inputs, `wrank` and `srank` are (N,) numpy
  arrays containing the rank of the true class in the output of the weak and
  strong classifier respectively. It returns an (N,) array of "loss" or
  "penalty" values based on `ctype`: top-1 error if `ctype=0`, top-5 error if
  `ctype=1`, and rank trucated by 10 if `ctype=2`.
  
- `calib(logits, gtlbl)`: For the logits of N training set inputs produced by a
  classifier over C classes, `logits` is an (N,C) numpy array containing the
  logits, and `gtlbl` is an (N,) integer array containing the class number
  (0-indexed) of the true class. This performs a calibration step to returns a
  single floating point number which, when multiplied with the logits, minimizes
  the cross-entropy loss.
  
- `entropy(logits, tinv)`: Computes entropy for an (N,C) numpy array of logits
  to return an (N,) array of entropy values. Call with `tinv` equal to the
  output of `calib`, or to 1.0 if you wish to compute entropy with respect to
  the original logits.
  
- `getqpm(rate, bdepth)`: Given rational rate and bucket depth, returns a tuple
  (Q,P,M) of integers such that rate=Q/P and bdepth=M/P. Approximates if
  necessary so that `P <= 100`.
  
- `getvpidx(rate, bpdeth)`: Utility function to return the token counts
  corresponding to indices in policy and token state vectors returned by this
  library.
  
# policy.py

- `fitmetric(etrain, rtrue)`: Call with a training set of N entropy values
  `h(x)` and true rewards `R(x,y)` as (N,) numpy arrays `etrain` and `rtrue`.
  Finds the mapping `f(h)` from entropy to metric, and returns this as tuple
  representing as a lookup table. Given this returned tuple `f`, you can map a
  new vector `enew` of entropy values to corresponding metrics by calling
  `np.interp(enew, *f)`.

- `mdp(rate, bdepth, traindata)` (see code for other optional parameters):
  Computes an optimal policy for a token bucket with a given rate and bucket
  depth, based on a training set `traindata = (metrics, rewards)`, where
  `metrics` and `rewards` are both (N,) dimensional numpy arrays containing
  containing corresponding metric and reward values of N training samples.
  Returns the policy vector of thresholds.

# simulate.py

- `simulate(rate, bdepth, policy, dset_mr)`: Simulate sending inputs with
  `policy` (a vector of thresholds as returned by the `mdp` function) for a
  token bucket with parameters `rate` and `bdepth`. Inputs are sampled i.i.d.
  from the dataset `dset_mr`, which is a tuple of corresponding metric values
  and rewards. By default, the simulation averages results over 100 streams each
  of length equal to 100k inputs sampled from this dataset (see the function for
  an optional argument to change these defaults). The code returns two outputs:
  the average reward attained by the policy, and a three tuple of `(send_m,
  send_s, occup_s)` for visualization. 
      - `send_m` is a (N,2) array, where N is the length of the dataset. The
        first dimension in `send_m` are sorted by metric values. `send_m[:,1]`
        contains the count of how many times that particular input was included
        in the stream, and `send_m[:,0]` the number of times it was actually
        offloaded.
      - `send_s` is a vector representing how often an input was offloaded in
        different token bucket states (with different number of tokens left).
      - `occup_s` is a vector representing the probability distribution of
        number of tokens left in the bucket while the policy is in operation.

- `mcsimulate(rb_i, rb_g, ncam, policy, dset_mr)`: Simulate sending inputs by
  multiple devices according to (the same) local policies, while sharing a
  network access switch, which enforces its own token bucket constraints. Here,
  `rb_i` is tuple of the `(rate_i, bdepth_i)` parameters for the token bucket
  running on each device, and `policy` the corresponding policy threshold
  vector. `rb_g` is a tuple with the global token bucket parameters, and `ncam`
  the number of devices / cameras. `dset_mr` is the dataset of examples for
  doing the simulation like for `simulate`: a tuple of vectors of the metric and
  reward values. The function has two outputs: the average reward, and the
  probability distribution of number of tokens left in the bucket at the
  access switch.<br /> <br />
  *Note that in `rb_g`, the rate component passed as input should be r_tot x ncam
  for the definition of r_tot used in the paper which is with respect to
  per-device input arrival frequency, while this function assumes network switch
  rate is defined in terms of arrivals at the switch (which is ncam times the
  arrival at each device). See usage in the test scripts.*
