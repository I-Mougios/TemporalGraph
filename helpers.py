from itertools import islice, count
from collections import namedtuple

Edge = namedtuple('Edge', 'source target timestamp')


def clean_edge(file):
    for row in file:
        clean_row = row.strip('\n').split()
        source, target, timestamp = clean_row
        yield Edge(source, target, int(timestamp))


def open_file(filepath, mode='r', n_rows=None):
    with open(filepath, mode) as f:
        f = islice(f, n_rows)
        yield from clean_edge(f)


def create_evenly_spaced_time_periods(start: int, stop: int, n_periods: int):
    entire_time_interval = (stop - start)
    time_period_interval = entire_time_interval / n_periods
    # N time periods require N+1 distinct timestamps
    timestamps = list(islice(count(start, step=time_period_interval), n_periods + 1))

    return [(round(timestamps[i]), round(timestamps[i + 1])) for i in range(n_periods)]


