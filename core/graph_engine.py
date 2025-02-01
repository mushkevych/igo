import asyncio
import logging
import time
from asyncio import Event, Task, create_task, gather, iscoroutinefunction, to_thread
from typing import Any, Callable, Coroutine, Optional, Concatenate, ParamSpec, Self

from core.feature_flag import FeatureFlagMgr

logger = logging.getLogger(__name__)

# Define a parameter spec to capture additional positional and named arguments, available later via P.args and P.kwargs
P = ParamSpec('P')

# Serial function type (Synchronous)
SeFunctionType = Callable[Concatenate[dict[str, Any], dict[str, Any], P], dict[str, Any]]

# Coroutine function type (Asynchronous)
CoFunctionType = Callable[Concatenate[dict[str, Any], dict[str, Any], P], Coroutine[Any, Any, dict[str, Any]]]


class ComputationalNode:
    def __init__(
        self,
        name: str,
        parents: Optional[list],
        feature_flag: Optional[str],
        exe_function: Optional[SeFunctionType] | Optional[CoFunctionType],
        exe_kwargs: Optional[dict[str, Any]] = None,
        exe_condition: Optional[Callable[[dict[str, Any], dict[str, Any]], bool]] = None,
    ) -> None:
        if exe_kwargs is None:
            exe_kwargs = {}
        if parents is None:
            parents = []

        self.name = name
        self.parents = parents
        self.feature_flag = feature_flag
        self.exe_function = exe_function
        self.exe_kwargs = exe_kwargs
        self.exe_condition = exe_condition
        self.children = set()

        for parent in self.parents:
            parent.add_child(self)

        self._event = None

    def __hash__(self) -> int:
        return hash(f'{self.name}::{self.feature_flag}')

    def __str__(self) -> str:
        return f'Node(name={self.name}, parents={self.parents}, feature_flag={self.feature_flag})'

    @property
    def event(self) -> Event:
        if not self._event:
            self._event = Event()
        return self._event

    def add_child(self, node: Self) -> Self:
        self.children.add(node)
        return self

    async def run(
        self,
        record: dict[str, Any],
        output: dict[str, Any],
        feature_flag_manager: FeatureFlagMgr,
    ) -> dict[str, str]:
        """
        Performs the inference orchestration for a given input record.

        This method asynchronously processes an immutable `record` and updates 
        the `output` dictionary in place inside `exe_function` with the computed results. 
        Additionally, it utilizes the `feature_flag_manager` to check for dynamic configuration 
        changes via environment variables.

        :param record: An immutable dictionary representing the key-value data record to process.
        :param output: A mutable dictionary for storing results, modified in-place by the `self.exe_function`.
        :param feature_flag_manager: A background service that monitors environment variables for feature flag updates.
        :return: A dictionary containing performance metrics collected during execution.
        """
        logger.debug(f'Executing node {self.name}')
        perf_stats: dict[str, str] = dict()
        try:
            t0 = time.time()
            if self.parents:
                # wait for all parents to complete
                await gather(*[parent.event.wait() for parent in self.parents])
            t_awaiting_parents = time.time() - t0

            t1 = time.time()
            if self.feature_flag:
                if not feature_flag_manager.is_enabled(self.feature_flag):
                    logger.debug(f'{self.name}.feature_flag disabled, skipping')
                    perf_stats[f'{self.name}.feature_flag'] = 'False'
                    return perf_stats

            if self.exe_condition:
                if not self.exe_condition(record, output):
                    logger.debug(f'{self.name}.exe_condition not satisfied, skipping')
                    perf_stats[f'{self.name}.exe_condition'] = 'False'
                    return perf_stats

            if self.exe_function:
                logger.debug(f'{self.name}.exe_condition satisfied, executing task')
                if iscoroutinefunction(self.exe_function):
                    _ = await create_task(self.exe_function(record, output, **self.exe_kwargs))
                else:
                    _ = await to_thread(self.exe_function, record, output, **self.exe_kwargs)

            perf_stats[f'{self.name}.awaiting_parents'] = f'{t_awaiting_parents:.3f}'
            perf_stats[f'{self.name}.execution'] = f'{time.time() - t1:.3f}'
            return perf_stats

        except Exception as e:
            perf_stats[f'{self.name}.exception'] = str(e)
            return perf_stats

        finally:
            # notify child Nodes that their parent Node has completed execution
            logger.debug(f'Executing node {self.name} DONE')
            self.event.set()


class ComputationalGraph:
    def __init__(self) -> None:
        self.root = ComputationalNode(name='root', parents=None, feature_flag=None, exe_function=None)

    async def run(self, record: dict[str, Any], feature_flag_manager: FeatureFlagMgr) -> tuple[dict[str, Any], dict[str, Any]]:
        output: dict[str, Any] = dict()             # A dictionary of results
        visited: list[ComputationalNode] = list()   # A list of visited nodes
        queue: list[ComputationalNode] = list()     # A queue of 'parent nodes' to process their children
        t0 = time.time()

        def bfs(node: ComputationalNode) -> list[Task]:
            logger.debug(f'Starting with node {node.name}')
            visited.append(node)
            queue.append(node)
            tasks: list[Task] = []

            while queue:
                n = queue.pop(0)

                for a_child in n.children:
                    if a_child not in visited:
                        assert isinstance(a_child, ComputationalNode)
                        visited.append(a_child)
                        queue.append(a_child)

                        # `output` dict is updated `in-place` inside every node
                        logger.debug(f'Scheduling child node {a_child.name}')
                        task: Task = asyncio.create_task(a_child.run(record, output, feature_flag_manager))
                        tasks.append(task)

            return tasks

        tasks = bfs(self.root)
        measurements: list[dict[str, Any]] = await asyncio.gather(*tasks)

        perf_stats: dict[str, Any] = dict()  # dictionary of performance metrics taken during Graph execution
        for measurement in measurements:
            perf_stats.update(measurement)

        perf_stats['dag.execution'] = f'{time.time() - t0:.3f}'
        return perf_stats, output
