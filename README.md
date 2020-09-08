# bar-ica-test

Trying to replicate the bar problem from Triesch 2007, "Synergies Between Intrinsic and Synaptic PlasticityMechanisms".

To preview, follow: <https://htmlpreview.github.io/?https://github.com/tuanpham96/bar-ica-test/blob/master/html/replicate_barprob_triesch2007IPandSP.html>

Main implementation in [`replicate_barprob_triesch2007IPandSP.m`](/replicate_barprob_triesch2007IPandSP.m).

Currently could not replicate bar detection. Either SP alone, or SP+IP (tried a few learning rates between `1E-4` to `1E-1`) would result in a roughly normal weight vector distribution, not a bimodal distribution like the paper. Also tried playing around with target rate `mu = 0.01, 0.05, 0.1`, bar probability `p_bar = 1/N, 1/2N` and initial activation parameters `a_init, b_init` - did not matter.

Possible issues might arise from:

- Implementation of IP or Hebbian SP
- How inputs are generated, whether the way to normalize is correct
- How weights are initialized. Currently, drawn from a uniform distribution [0, 1], then normalized to unit length. Additionally, also tried a normal distribution, a log-normal distribution and a few beta distributions (didn't work).
