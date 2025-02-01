import asyncio
import json
import time
import unittest
from typing import Any

from core.feature_flag import FeatureFlagMgr
from core.graph_engine import ComputationalGraph, ComputationalNode

flag_manager = FeatureFlagMgr()


def display_metrics(stats: dict[str, Any]):
    print(json.dumps(stats, indent=2))


async def async_task_executor(
    input_data: dict[str, Any], results: dict[str, Any], identifier: str, msg: str, start_time: float,
    delay: float = 0.0
) -> dict[str, Any]:
    results[identifier] = msg
    elapsed = time.time() - start_time
    print(f'started_at={elapsed:.3f} \t expected_finish={elapsed + delay:.3f} \t {input_data}')
    await asyncio.sleep(delay)
    return input_data


def sync_task_executor(
    input_data: dict[str, Any], results: dict[str, Any], identifier: str, msg: str, start_time: float,
    delay: float = 0.0
) -> dict[str, Any]:
    results[identifier] = msg
    elapsed = time.time() - start_time
    print(f'started_at={elapsed:.3f} \t expected_finish={elapsed + delay:.3f} \t {input_data}')
    time.sleep(delay)
    return input_data


def error_task_executor(
    input_data: dict[str, Any], results: dict[str, Any], identifier: str, msg: str, start_time: float,
    delay: float = 0.0
) -> dict[str, Any]:
    raise ValueError('Intentional error occurred')


class GraphEngineTestSuite(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_start = time.time()

    async def test_basic_node(self) -> None:
        node = ComputationalNode(
            'basic_node',
            parents=[],
            feature_flag=None,
            exe_function=sync_task_executor,
            exe_kwargs={
                'identifier': 'basic_node',
                'msg': 'test node',
                'start_time': self.test_start,
                'delay': 0.02
            },
        )

        self.assertFalse(node.event.is_set())
        input_data = {'key1': 'value1', 'key2': 'value2'}
        output = {}
        metrics = await node.run(record=input_data, output=output, feature_flag_manager=flag_manager)
        display_metrics(metrics)
        self.assertIn('basic_node', output)
        self.assertTrue(node.event.is_set())

    async def test_node_failure(self) -> None:
        node = ComputationalNode(
            'error_node',
            parents=[],
            feature_flag=None,
            exe_function=error_task_executor,
            exe_kwargs={
                'identifier': 'error_node',
                'msg': 'failure test',
                'start_time': self.test_start,
                'delay': 0.02
            },
        )

        self.assertFalse(node.event.is_set())
        input_data = {'key1': 'value1', 'key2': 'value2'}
        output = {}
        metrics = await node.run(record=input_data, output=output, feature_flag_manager=flag_manager)
        display_metrics(metrics)
        self.assertIn(f'{node.name}.exception', metrics)
        self.assertTrue(node.event.is_set())

    async def test_linear_graph_execution(self) -> None:
        graph = ComputationalGraph()
        node1 = ComputationalNode(
            'start_node',
            parents=[],
            feature_flag=None,
            exe_function=sync_task_executor,
            exe_kwargs={
                'identifier': 'start_node',
                'msg': 'initiator',
                'start_time': self.test_start,
                'delay': 0.02
            },
        )
        node2 = ComputationalNode(
            'middle_node',
            parents=[node1],
            feature_flag=None,
            exe_function=sync_task_executor,
            exe_kwargs={
                'identifier': 'middle_node',
                'msg': 'progressing',
                'start_time': self.test_start,
                'delay': 0.05
            },
        )
        node3 = ComputationalNode(
            'end_node',
            parents=[node2],
            feature_flag=None,
            exe_function=sync_task_executor,
            exe_kwargs={
                'identifier': 'end_node',
                'msg': 'completion',
                'start_time': self.test_start,
                'delay': 0.02
            },
        )

        graph.root.add_child(node1)

        input_data = {'key1': 'value1', 'key2': 'value2'}
        metrics, output = await graph.run(record=input_data, feature_flag_manager=flag_manager)
        display_metrics(metrics)
        display_metrics(output)

        for node_name in ['start_node', 'middle_node', 'end_node']:
            self.assertIn(node_name, output)
        self.assertTrue(node1.event.is_set())
        self.assertTrue(node2.event.is_set())
        self.assertTrue(node3.event.is_set())

    async def test_conditional_execution(self) -> None:
        node = ComputationalNode(
            'conditional_node',
            parents=[],
            feature_flag=None,
            exe_function=sync_task_executor,
            exe_kwargs={
                'identifier': 'conditional_node',
                'msg': 'conditional execution',
                'start_time': self.test_start,
                'delay': 0.02
            },
            exe_condition=lambda data, _: 'trigger_key' in data,
        )

        input_data = {'key1': 'value1'}
        output = {}
        metrics = await node.run(record=input_data, output=output, feature_flag_manager=flag_manager)
        display_metrics(metrics)
        self.assertNotIn('conditional_node', output)

        input_data['trigger_key'] = 'activate'
        metrics = await node.run(record=input_data, output=output, feature_flag_manager=flag_manager)
        display_metrics(metrics)
        self.assertIn('conditional_node', output)

    async def test_pyramid_structure(self) -> None:
        graph = ComputationalGraph()
        base_nodes = [
            ComputationalNode(
                f'base_node_{i}',
                parents=[],
                feature_flag=None,
                exe_function=sync_task_executor,
                exe_kwargs={
                    'identifier': f'base_node_{i}',
                    'msg': f'Base {i}',
                    'start_time': self.test_start,
                    'delay': 0.1 * (3 - i)
                },
            ) for i in range(3)
        ]
        middle_node = ComputationalNode(
            'middle_layer',
            parents=base_nodes,
            feature_flag=None,
            exe_function=sync_task_executor,
            exe_kwargs={
                'identifier': 'middle_layer',
                'msg': 'Aggregated middle',
                'start_time': self.test_start,
                'delay': 0.2
            },
        )
        top_node = ComputationalNode(
            'top_node',
            parents=[middle_node],
            feature_flag=None,
            exe_function=sync_task_executor,
            exe_kwargs={
                'identifier': 'top_node',
                'msg': 'Final aggregation',
                'start_time': self.test_start,
                'delay': 0.1
            },
        )

        for node in base_nodes:
            graph.root.add_child(node)

        input_data = {'key1': 'value1'}
        metrics, output = await graph.run(record=input_data, feature_flag_manager=flag_manager)
        display_metrics(metrics)
        display_metrics(output)

        for node in base_nodes + [middle_node, top_node]:
            self.assertIn(node.name, output)
            self.assertTrue(node.event.is_set())


if __name__ == '__main__':
    unittest.main()
