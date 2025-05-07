# Codes from https://github.com/facebookresearch/SimulEval

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from statistics import mean
import logging

from typing import Dict, List, Union

logger = logging.getLogger(__name__)

LATENCY_SCORERS_DICT = {}
LATENCY_SCORERS_NAME_DICT = {}


class Instance:
    def __init__(self, interval: float, reference: list=[]) -> None:
        self.interval = interval
        self.reference = reference
        self.reference_length = len(reference)
        self.delays = []
        self.token_chunk_id = []
        self.source_length = 0
        self.metrics = {}
    
    def append_segment(self, tokens: list, segment_id: int, compute_time: float = 0.0):
        if type(tokens) == int:
            tokens = [tokens]
        segment_delay = [(segment_id+1)*self.interval+compute_time]*len(tokens)
        self.delays += segment_delay
        self.source_length += self.interval
        for _ in tokens:
            self.token_chunk_id.append(segment_id)


class LatencyScorer:
    metric = None
    add_duration = False

    def __init__(
        self, computation_aware: bool = False, use_ref_len: bool = True
    ) -> None:
        super().__init__()
        self.use_ref_len = use_ref_len
        self.computation_aware = computation_aware

    @property
    def timestamp_type(self):
        return "delays" if not self.computation_aware else "elapsed"

    def compute(self, *args):
        raise NotImplementedError

    def get_delays_lengths(self, ins: Instance):
        """
        Args:
            ins Instance: one instance

        Returns:
            A tuple with the 3 elements:
            delays (List[Union[float, int]]): Sequence of delays.
            src_len (Union[float, int]): Length of source sequence.
            tgt_len (Union[float, int]): Length of target sequence.
        """
        delays = getattr(ins, self.timestamp_type, None)
        assert delays

        if not self.use_ref_len or ins.reference is None:
            tgt_len = len(delays)
        else:
            tgt_len = ins.reference_length
        src_len = ins.source_length
        return delays, src_len, tgt_len
    
    def get_chunk_ids(self, ins: Instance):
        if len(ins.token_chunk_id) == len(ins.delays):
            return ins.token_chunk_id
        else:
            raise ValueError("Instance has no chunk id information")

    @property
    def metric_name(self) -> str:
        return self.__class__.__name__

    def __call__(self, instances: Dict[int, Instance]) -> float:
        scores = []
        for index, ins in instances.items():
            delays = getattr(ins, self.timestamp_type, None)
            if delays is None or len(delays) == 0:
                logger.warn(f"Instance {index} has no delay information. Skipped")
                continue
            score = self.compute(ins)
            ins.metrics[self.metric_name] = score
            scores.append(score)

        return mean(scores)


class ALScorer(LatencyScorer):
    r"""
    Average Lagging (AL) from
    `STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency using Prefix-to-Prefix Framework <https://arxiv.org/abs/1810.08398>`_

    Give source :math:`X`, target :math:`Y`, delays :math:`D`,

    .. math::

        AL = \frac{1}{\tau} \sum_i^\tau D_i - (i - 1) \frac{|X|}{|Y|}

    Where

    .. math::

        \tau = argmin_i(D_i = |X|)

    When reference was given, :math:`|Y|` would be the reference length

    Usage:
        ----latency-metrics AL
    """  # noqa: E501

    def compute(self, ins: Instance):
        """
        Function to compute latency on one sentence (instance).

        Args:
            ins Instance: one instance

        Returns:
            float: the latency score on one sentence.
        """
        delays, source_length, target_length = self.get_delays_lengths(ins)

        if delays[0] > source_length:
            return delays[0]

        AL = 0
        gamma = target_length / source_length
        tau = 0
        for t_minus_1, d in enumerate(delays):
            AL += d - t_minus_1 / gamma
            tau = t_minus_1 + 1

            if d >= source_length:
                break
        AL /= tau
        return AL


class LAALScorer(ALScorer):
    r"""
    Length Adaptive Average Lagging (LAAL) as proposed in
    `CUNI-KIT System for Simultaneous Speech Translation Task at IWSLT 2022
    <https://arxiv.org/abs/2204.06028>`_.
    The name was suggested in `Over-Generation Cannot Be Rewarded:
    Length-Adaptive Average Lagging for Simultaneous Speech Translation
    <https://arxiv.org/abs/2206.05807>`_.
    It is the original Average Lagging as proposed in
    `Controllable Latency using Prefix-to-Prefix Framework
    <https://arxiv.org/abs/1810.08398>`_
    but is robust to the length difference between the hypothesis and reference.

    Give source :math:`X`, target :math:`Y`, delays :math:`D`,

    .. math::

        LAAL = \frac{1}{\tau} \sum_i^\tau D_i - (i - 1) \frac{|X|}{max(|Y|,|Y*|)}

    Where

    .. math::

        \tau = argmin_i(D_i = |X|)

    When reference was given, :math:`|Y|` would be the reference length, and :math:`|Y*|` is the length of the hypothesis.

    Usage:
        ----latency-metrics LAAL
    """

    def compute(self, ins: Instance):
        """
        Function to compute latency on one sentence (instance).

        Args:
            ins: Instance: one instance

        Returns:
            float: the latency score on one sentence.
        """
        delays, source_length, target_length = self.get_delays_lengths(ins)
        if delays[0] > source_length:
            return delays[0]

        LAAL = 0
        gamma = max(len(delays), target_length) / source_length
        tau = 0
        for t_minus_1, d in enumerate(delays):
            LAAL += d - t_minus_1 / gamma
            # print("AL", LAAL, d, t_minus_1 / gamma)
            tau = t_minus_1 + 1

            if d >= source_length:
                break
        LAAL /= tau
        return LAAL



class APScorer(LatencyScorer):
    r"""
    Average Proportion (AP) from
    `Can neural machine translation do simultaneous translation? <https://arxiv.org/abs/1606.02012>`_

    Give source :math:`X`, target :math:`Y`, delays :math:`D`,
    the AP is calculated as:

    .. math::

        AP = \frac{1}{|X||Y]} \sum_i^{|Y|} D_i

    Usage:
        ----latency-metrics AP
    """

    def compute(self, ins: Instance) -> float:
        """
        Function to compute latency on one sentence (instance).

        Args:
            ins Instance: one instance

        Returns:
            float: the latency score on one sentence.
        """
        delays, source_length, target_length = self.get_delays_lengths(ins)
        return sum(delays) / (source_length * target_length)



class DALScorer(LatencyScorer):
    r"""
    Differentiable Average Lagging (DAL) from
    Monotonic Infinite Lookback Attention for Simultaneous Machine Translation
    (https://arxiv.org/abs/1906.05218)

    Usage:
        ----latency-metrics DAL
    """

    def compute(self, ins: Instance):
        """
        Function to compute latency on one sentence (instance).

        Args:
            ins Instance: one instance

        Returns:
            float: the latency score on one sentence.
        """
        delays, source_length, target_length = self.get_delays_lengths(ins)

        DAL = 0
        target_length = len(delays)
        gamma = target_length / source_length
        g_prime_last = 0
        for i_minus_1, g in enumerate(delays):
            if i_minus_1 + 1 == 1:
                g_prime = g
            else:
                g_prime = max([g, g_prime_last + 1 / gamma])

            DAL += g_prime - i_minus_1 / gamma
            g_prime_last = g_prime

        DAL /= target_length
        return DAL
    

class NumChunksScorer(LatencyScorer):
    """Number of chunks (of speech/text) in output

    Usage:
        ----latency-metrics NumChunks

    """

    def compute(self, ins: Instance):
        delays, _, _ = self.get_delays_lengths(ins)
        return len(delays)


class RTFScorer(LatencyScorer):
    """Compute Real Time Factor (RTF)

    Usage:
        ----latency-metrics (RTF)

    """

    def compute(self, ins: Instance):
        delays, source_length, _ = self.get_delays_lengths(ins)
        delays = [start + duration for start, duration in ins.intervals]
        return delays[-1] / source_length
    

class ATDScorer(LatencyScorer):
    r"""
    Average Token Delay (ATD) from
    Average Token Delay: A Latency Metric for Simultaneous Translation
    (https://arxiv.org/abs/2211.13173)

    Different from speech segments, text tokens have no length
    and multiple tokens can be output at the same time like subtitle.
    Therefore, we set its length to be 0. However, to calculate latency in text-text,
    we give virtual time 1 for the length of text tokens.

    Usage:
        ----latency-metrics ATD
    """

    def __call__(self, instances) -> float:  # noqa C901
        # SRC_TOKEN_LEN =  0.3  # 300ms per word
        # INPUT_TYPE = "speech"
        TGT_TOKEN_LEN = 0
        # OUTPUT_TYPE = "text"

        scores = []
        for index, ins in instances.items():

            SRC_TOKEN_LEN = ins.source_length / ins.reference_length
            # print(SRC_TOKEN_LEN)

            delays = getattr(ins, "delays", None)
            if delays is None or len(delays) == 0:
                logger.warn(f"Instance {index} has no delay information. Skipped")
                continue

            if self.computation_aware:
                elapsed = getattr(ins, "elapsed", None)
                if elapsed is None or len(elapsed) == 0:
                    logger.warn(
                        f"Instance {index} has no computational delay information. Skipped"
                    )
                    continue
                if elapsed != [0] * len(delays):
                    compute_elapsed = self.subtract(elapsed, delays)
                    compute_times = self.subtract(
                        compute_elapsed, [0] + compute_elapsed[:-1]
                    )
                else:
                    compute_times = elapsed
            else:
                compute_times = [0] * len(delays)

            chunk_sizes = {"src": [0], "tgt": [0]}
            token_to_chunk = {"src": [0], "tgt": [0]}
            token_to_time = {"src": [0], "tgt": [0]}

            tgt_token_lens = []
            delays_no_duplicate = sorted(set(delays), key=delays.index)

            prev_delay = None
            for delay in delays:
                if delay != prev_delay:
                    chunk_sizes["tgt"].append(1)
                else:
                    chunk_sizes["tgt"][-1] += 1
                prev_delay = delay
            for i, chunk_size in enumerate(chunk_sizes["tgt"][1:], 1):
                token_to_chunk["tgt"] += [i] * chunk_size
            tgt_token_lens = [TGT_TOKEN_LEN] * len(delays)

            chunk_durations = self.subtract(
                delays_no_duplicate, [0] + delays_no_duplicate[:-1]
            )
            for i, chunk_duration in enumerate(chunk_durations, 1):
                num_tokens, rest = divmod(chunk_duration, SRC_TOKEN_LEN)
                token_lens = int(num_tokens) * [SRC_TOKEN_LEN] + (
                    [rest] if rest != 0 else []
                )
                chunk_sizes["src"] += [len(token_lens)]
                for token_len in token_lens:
                    token_to_time["src"].append(
                        token_to_time["src"][-1] + token_len
                    )
                    token_to_chunk["src"].append(i)

            for delay, compute_time, token_len in zip(
                delays, compute_times, tgt_token_lens
            ):
                tgt_start_time = max(delay, token_to_time["tgt"][-1])
                token_to_time["tgt"].append(tgt_start_time + token_len + compute_time)
            print(token_to_time["src"])
            scores.append(self.compute(chunk_sizes, token_to_chunk, token_to_time))

        return mean(scores)

    def subtract(self, arr1, arr2):
        return [x - y for x, y in zip(arr1, arr2)]

    def compute(
        self,
        chunk_sizes: Dict[str, List[Union[float, int]]],
        token_to_chunk: Dict[str, List[Union[float, int]]],
        token_to_time: Dict[str, List[Union[float, int]]],
    ) -> float:
        """
        Function to compute latency on one sentence (instance).
        Args:
            chunk_sizes Dict[str, List[Union[float, int]]]: Sequence of chunk sizes for source and target.
            token_to_chunk Dict[str, List[Union[float, int]]]: Sequence of chunk indices to which the tokens belong for source and target.
            token_to_time Dict[str, List[Union[float, int]]]: Sequence of ending times of tokens for source and target.

        Returns:
            float: the latency score on one sentence.
        """  # noqa C501

        tgt_to_src = []

        for t in range(1, len(token_to_chunk["tgt"])):
            chunk_id = token_to_chunk["tgt"][t]
            AccSize_x = sum(chunk_sizes["src"][:chunk_id])
            AccSize_y = sum(chunk_sizes["tgt"][:chunk_id])

            S = t - max(0, AccSize_y - AccSize_x)
            current_src_size = sum(chunk_sizes["src"][: chunk_id + 1])

            if S < current_src_size:
                tgt_to_src.append((t, S))
            else:
                tgt_to_src.append((t, current_src_size))

        atd_delays = []

        for t, s in tgt_to_src:
            atd_delay = token_to_time["tgt"][t] - token_to_time["src"][s]
            atd_delays.append(atd_delay)

        return float(mean(atd_delays))