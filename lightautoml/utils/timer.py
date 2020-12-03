"""
Timer
"""

from time import time
from typing import Optional, List

import numpy as np
from log_calls import record_history

from .logging import get_logger, DuplicateFilter

logger = get_logger(__name__)
logger.addFilter(DuplicateFilter())


@record_history(enabled=False)
class Timer:
    _timeout = 1e10
    _overhead = 0
    _mode = 1

    def __init__(self):
        self.start_time = None
        self.total_duration = None

    @property
    def time_left(self) -> float:
        return self.timeout - self.time_spent

    @property
    def time_spent(self) -> float:
        return time() - self.start_time

    @property
    def perc_left(self) -> float:
        return self.time_left / self.timeout

    @property
    def perc_spent(self) -> float:
        return self.time_spent / self.timeout

    @property
    def timeout(self) -> float:
        return self._timeout

    def time_limit_exceeded(self) -> bool:
        if self._mode == 0:
            return False

        if self._mode == 1:
            return self.time_left < 0

        if self._mode == 2:
            return (self.time_left - self._overhead) < 0

    def start(self):
        self.start_time = time()
        return self

    def stop(self):
        self.total_duration = time() - self.start_time


@record_history(enabled=False)
class PipelineTimer(Timer):
    """
    Timer is used to control time over full automl run.
    It decides how much time spend to each algo
    """

    def __init__(self, timeout: Optional[float] = None, overhead: float = .1, mode: int = 1, tuning_rate: float = 0.7):
        """
        Create global automl timer

        Args:
            timeout:
            overhead: (0, 1) - rate of time that will be used to early stop.
                Ex. if set to 0.1 and timing mode is set to 2, timer will finish tasks after 0.9 of all time spent
            mode: Timing mode
                0 - timer is used to estimate runtime but if something goes out of time - keep it run (Real life mode)
                1 - timer is used to terminate tasks, but do it after real timeout (Trade off mode)
                2 - timer is used to terminate tasks with the goal to be exactly in time (Benchmarking/competitions mode)
                Keep in mind - all time limitations will turn on after at least single model/single fold will be computed
            tuning_rate:
        """
        if timeout is not None:
            self._timeout = timeout
        self._task_scores = 0
        self._rate_overhead = overhead
        self._overhead = overhead * self.timeout
        self.run_info = {}
        self._mode = mode
        self.tuning_rate = tuning_rate
        self.child_out_of_time = False

    def add_task(self, score: float = 1.0):
        self._task_scores += score

    def close_task(self, score: float = 1.0):
        self._task_scores -= score

    def get_time_for_next_task(self, score: float = 1.0):
        if round(self._task_scores, 3) == 0:
            return self.time_left

        return (self.time_left - self._overhead) * (score / self._task_scores)

    def get_task_timer(self, key: Optional[str] = None, score: float = 1.0) -> 'TaskTimer':
        return TaskTimer(self, key, score, self._rate_overhead, self._mode, self.tuning_rate)


@record_history(enabled=False)
class TaskTimer(Timer):
    """
    Timer is used to control time over single ML task run.
    It decides how much time is ok to spend on tuner and if we have enough time to calc more folds
    """

    @property
    def in_progress(self) -> bool:
        return self.start_time is not None

    def __init__(self, pipe_timer: PipelineTimer, key: Optional[str] = None, score: float = 1.0,
                 overhead: Optional[float] = 1, mode: int = 1,
                 default_tuner_time_rate: float = 0.7):
        """


        Args:
            pipe_timer: global automl timer
            key: string name that will be associated with this task
            score: time score for current task. Defaults to 1.0. For ex. if you want to give more of total time to task set it > 1
            overhead: see overhead of PipelineTimer
            mode: see mode for PipelineTimer
            default_tuner_time_rate: if no timing history for the moment of estimating tuning time,
                timer will use this rate of time_left
        """
        self.score = score
        pipe_timer.add_task(self.score)
        self.pipe_timer = pipe_timer
        self.start_time = None
        self.key = key
        self._rate_overhead = overhead
        self._mode = mode
        self.default_tuner_rate = default_tuner_time_rate

    def start(self):
        if self.in_progress:
            return self

        self.start_time = time()
        self._timeout = self.pipe_timer.get_time_for_next_task(self.score)
        self._overhead = self._rate_overhead * self.time_left
        self.pipe_timer.close_task(self.score)

        return self

    def set_control_point(self):
        self._timeout = self.timeout - self.time_spent
        self.start_time = time()

    def write_run_info(self):

        if self.key in self.pipe_timer.run_info:
            self.pipe_timer.run_info[self.key].append(self.time_spent)
        else:
            self.pipe_timer.run_info[self.key] = [self.time_spent]

    def get_run_results(self) -> Optional[np.ndarray]:

        if self.key in self.pipe_timer.run_info:
            return self.pipe_timer.run_info[self.key]
        else:
            return None

    def estimate_folds_time(self, n_folds: int = 1) -> Optional[float]:
        """
        Estimate time for n_folds

        Args:
            n_folds:

        Returns:

        """
        run_results = self.get_run_results()
        if run_results is None:
            return None

        if self._mode > 0:
            single_run_est = np.max(run_results)
        else:
            single_run_est = run_results.mean()

        return single_run_est * n_folds

    def estimate_tuner_time(self, n_folds: int = 1) -> float:
        """
        Estimates time that is ok to spend on tuner

        Returns:

        """
        folds_est = self.estimate_folds_time(n_folds)
        if folds_est is None:
            return self.default_tuner_rate * self.time_left
        return self.time_left - folds_est

    def time_limit_exceeded(self) -> bool:
        """
        Estimate time limit and send results to parent timer.

        Returns:

        """
        out_of_time = super().time_limit_exceeded()
        if out_of_time:
            self.pipe_timer.child_out_of_time = True
        return out_of_time

    def __copy__(self):

        proxy_timer = PipelineTimer().start()
        logger.warning('Copying TaskTimer may affect the parent PipelineTimer, so copy will create new unlimited TaskTimer')

        return proxy_timer.get_task_timer(self.key)

    def __deepcopy__(self, *args, **kwargs):

        return self.__copy__()

    def split_timer(self, n_parts: int) -> List['TaskTimer']:

        new_tasks_score = self.score / n_parts
        timers = [self.pipe_timer.get_task_timer(self.key, new_tasks_score) for _ in range(n_parts)]
        self.pipe_timer.close_task(self.score)

        return timers
