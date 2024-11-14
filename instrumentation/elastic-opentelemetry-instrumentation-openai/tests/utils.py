# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Mapping, Optional, Sequence

from opentelemetry.sdk._logs._internal import LogData
from opentelemetry.sdk.metrics._internal.point import Metric
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
    DataPointT,
    HistogramDataPoint,
    NumberDataPoint,
)
from opentelemetry.util.types import AttributeValue


def get_sorted_metrics(memory_metrics_reader: InMemoryMetricReader):
    metrics_data = memory_metrics_reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics if metrics_data else []

    all_metrics = []
    for metrics in resource_metrics:
        for scope_metrics in metrics.scope_metrics:
            all_metrics.extend(scope_metrics.metrics)

    return sorted_metrics(all_metrics)


def sorted_metrics(metrics: Sequence[Metric]):
    """
    Sorts metrics by metric name.
    """
    return sorted(
        metrics,
        key=lambda m: m.name,
    )


def assert_metric_expected(
    metric: Metric,
    expected_data_points: Sequence[DataPointT],
    est_value_delta: Optional[float] = 0,
):
    assert len(expected_data_points) == len(metric.data.data_points)
    for expected_data_point in expected_data_points:
        assert_data_point_expected(expected_data_point, metric.data.data_points, est_value_delta)


def is_data_points_equal(
    expected_data_point: DataPointT,
    data_point: DataPointT,
    est_value_delta: Optional[float] = 0,
):
    if type(expected_data_point) != type(  # noqa: E721
        data_point
    ) or not isinstance(expected_data_point, (HistogramDataPoint, NumberDataPoint)):
        return False

    values_diff = None
    if isinstance(data_point, NumberDataPoint):
        values_diff = abs(expected_data_point.value - data_point.value)
    elif isinstance(data_point, HistogramDataPoint):
        values_diff = abs(expected_data_point.sum - data_point.sum)
        if expected_data_point.count != data_point.count or (
            est_value_delta == 0
            and (expected_data_point.min != data_point.min or expected_data_point.max != data_point.max)
        ):
            return False

    return values_diff <= est_value_delta and expected_data_point.attributes == dict(data_point.attributes)


def assert_data_point_expected(
    expected_data_point: DataPointT,
    data_points: Sequence[DataPointT],
    est_value_delta: Optional[float] = 0,
):
    is_data_point_exist = False
    for data_point in data_points:
        if is_data_points_equal(expected_data_point, data_point, est_value_delta):
            is_data_point_exist = True
            break

    if not is_data_point_exist:
        for data_point in data_points:
            print(data_point)
    assert is_data_point_exist is True, f"Data point {expected_data_point} does not exist."


def create_number_data_point(value, attributes):
    return NumberDataPoint(
        value=value,
        attributes=attributes,
        start_time_unix_nano=0,
        time_unix_nano=0,
    )


def create_histogram_data_point(sum_data_point, count, max_data_point, min_data_point, attributes):
    return HistogramDataPoint(
        count=count,
        sum=sum_data_point,
        min=min_data_point,
        max=max_data_point,
        attributes=attributes,
        start_time_unix_nano=0,
        time_unix_nano=0,
        bucket_counts=[],
        explicit_bounds=[],
    )


def logrecords_from_logs(logs: Sequence[LogData]) -> Sequence[Mapping[str, AttributeValue]]:
    return [log.log_record for log in logs]
