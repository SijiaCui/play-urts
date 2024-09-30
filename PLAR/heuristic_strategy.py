counter_coacAI_0_to_100 = """
START of TASK
[Harvest Mineral]((0, 0), (1, 0), (1, 2), (1, 1)),
[Harvest Mineral]((0, 0), (0, 1), (1, 2), (0, 2))],
[Produce Unit]('worker', 'north'),
[Produce Unit]('worker', 'south'),
[Deploy Unit]('worker', (2, 3)),
END of TASK
"""

counter_coacAI_100_to_300 = """
START of TASK
[Harvest Mineral]((0, 0), (1, 0), (1, 2), (1, 1)),
[Produce Unit]('worker', 'east'),
[Produce Unit]('worker', 'south'),
[Build Building]('barrack', (1, 4), (0, 4)),
[Deploy Unit]('worker', (2, 4)),
[Deploy Unit]('worker', (3, 0)),
[Deploy Unit]('worker', (4, 4)),
[Produce Unit]('ranged', 'east'),
[Produce Unit]('ranged', 'south'),
[Produce Unit]('ranged', 'east'),
[Produce Unit]('ranged', 'south'),
[Produce Unit]('light', 'south'),
[Produce Unit]('ranged', 'east'),
[Produce Unit]('ranged', 'south'),
[Produce Unit]('heavy', 'east'),
[Deploy Unit]('ranged', (3, 5)),
[Deploy Unit]('ranged', (4, 4)),
[Deploy Unit]('ranged', (4, 6))
END of TASK
"""

counter_coacAI_300_to_inf = """
START of TASK
[Harvest Mineral]((0, 0), (1, 0), (1, 2), (1, 1)),
[Produce Unit]('worker', 'east'),
[Produce Unit]('worker', 'south'),
[Harvest Mineral]((0, 0), (0, 1), (1, 2), (0, 2))],
[Deploy Unit]('worker', (2, 4)),
[Produce Unit]('ranged', 'east'),
[Produce Unit]('ranged', 'south'),
[Produce Unit]('ranged', 'east'),
[Produce Unit]('ranged', 'south'),
[Produce Unit]('light', 'south'),
[Produce Unit]('ranged', 'east'),
[Produce Unit]('ranged', 'south'),
[Produce Unit]('heavy', 'east'),
[Deploy Unit]('ranged', (3, 5)),
[Deploy Unit]('ranged', (4, 4)),
[Deploy Unit]('ranged', (4, 6))
END of TASK
"""
