# MIT License

# Copyright (c) 2024 Eleuther AI and The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This section of the code comes almost entirely from the harness
# We kept it because it's very fast - however, we renamed the variables
# and added documentation

import logging
import math
import os
import random
from typing import Callable, Optional

import numpy as np
from scipy.stats import bootstrap
from tqdm import tqdm


logger = logging.getLogger(__name__)

LIGHTEVAL_DISABLE_TQDM = int(os.environ.get("LIGHTEVAL_DISABLE_TQDM", "0"))


def _stddev(arr):
    mu = np.mean(arr)
    return math.sqrt(sum((x - mu) ** 2 for x in arr) / (len(arr) - 1))


def mean_stderr(arr):
    return _stddev(arr) / math.sqrt(len(arr))


SERIAL_MAX_SECONDS = 30.0      # a draw loop estimated above this is handed to a (bounded) process pool, as before
POOL_MAX_PROCESSES = 32        # the old pool had mp.cpu_count() (288 on a JUPITER node) processes, each a fork of the whole (tens of GB) lighteval process


class _bootstrap_internal:
    def __init__(self, number_draws: int, metric: Optional[Callable] = None):
        self.number_draws = number_draws
        self.metric = metric

    def __call__(self, cur_experiment):
        # Creates number_draws samplings (with replacement) of the population by iterating on a given seed.
        # (idle-time fix) The draws still come from ONE random.Random(seed) stream, consumed in the same order, and the metric of a draw does not depend on the other draws, so the numbers are
        # bit-identical to the old version, which started mp.Pool(mp.cpu_count()) for EVERY stderr (288 forks of a ~50 GB process: 27 s each, 46 stderrs = 21 min for one GSM-Infinite group).
        # Light metrics (a mean of a few thousand floats: 0.2 ms a draw) run in this process; a metric slow enough to need workers (an estimate from the first draws) uses a bounded pool.
        import time

        population, seed = cur_experiment
        rnd = random.Random()
        rnd.seed(seed)

        def draw():
            return rnd.choices(population, k=len(population))

        probe = min(5, self.number_draws)
        t0 = time.perf_counter()
        samplings = [self.metric(draw()) for _ in range(probe)]
        per_draw = (time.perf_counter() - t0) / max(probe, 1)
        rest = self.number_draws - probe
        if rest > 0 and per_draw * rest > SERIAL_MAX_SECONDS:
            import multiprocessing as mp

            with mp.Pool(min(mp.cpu_count(), POOL_MAX_PROCESSES)) as pool:
                samplings.extend(
                    pool.starmap(
                        self.metric,
                        tqdm(
                            [(draw(),) for _ in range(rest)],
                            total=rest,
                            desc="Sampling bootstrap iterations",
                            disable=bool(LIGHTEVAL_DISABLE_TQDM),
                        ),
                    )
                )
        else:
            samplings.extend(
                self.metric(draw())
                for _ in tqdm(
                    range(rest), total=rest, desc="Sampling bootstrap iterations", disable=bool(LIGHTEVAL_DISABLE_TQDM)
                )
            )
        return samplings


def bootstrap_stderr(metric: Callable, population: list, number_experiments: int):
    """Bootstraps the stderr of the given metric for the given population of samples,
    by sampling said population for number_experiments and recomputing the metric on the
    different samplings.
    """
    res = []
    number_draws = min(1000, number_experiments)
    number_seeds = number_experiments // number_draws

    logger.info(f"Bootstrapping {metric.__name__}'s stderr with {number_seeds} seeds.")
    for seed in range(number_seeds):
        # sample w replacement
        res.extend(_bootstrap_internal(metric=metric, number_draws=number_draws)((population, seed)))

    return _stddev(res)


def get_stderr_function(aggregation: Callable, number_experiments: int = 1000):
    # Mean stderr can be computed trivially
    if "mean" in aggregation.__name__:
        return mean_stderr

    # For other metrics, we bootstrap the stderr by sampling
    try:
        return lambda population: bootstrap_stderr(
            metric=aggregation, population=population, number_experiments=number_experiments
        )
    except Exception:
        return None


def bootstrap_stderr_scipy(metric: Callable, population: list, number_experiments: int = 1000):
    """Simulates resampling (draw with replacement, of the same size as the orig set) n times from the results population
    to compute the distance between the simulated resampling distribution and the actual distribution.
    Same as bootstrap_stderr, but uses scipy.
    It's kept for archive, as it overflows for big datasets
    """
    logger.info(f"Bootstrapping {metric.__name__}'s stderr.")
    res = bootstrap(
        data=[population],
        statistic=metric,
        n_resamples=number_experiments,
        confidence_level=0.95,
        method="BCa",
    )
    return res.standard_error
