counter_coacAI_0_to_200 = """
START of TASK
[Harvest Mineral](0, 0),
[Harvest Mineral](0, 0),
[Produce Unit]('worker', 'north'),
[Produce Unit]('worker', 'south'),
[Produce Unit]('worker', 'east'),
[Produce Unit]('worker', 'south'),
[Deploy Unit]('worker', (2, 3)),
END of TASK
"""

counter_coacAI_200_to_400 = """
START of TASK
[Build Building]('barrack', (0, 4)),
[Harvest Mineral](0, 0),
[Produce Unit]('worker', 'east'),
[Produce Unit]('worker', 'south'),
[Produce Unit]('worker', 'east'),
[Produce Unit]('worker', 'south'),
[Deploy Unit]('worker', (2, 3)),
END of TASK
"""

counter_coacAI_400_to_600 = """
START of TASK
[Harvest Mineral](0, 0),
[Deploy Unit]('worker', (2, 3)),
[Harvest Mineral](0, 0),
[Produce Unit]('worker', 'east'),
[Produce Unit]('worker', 'south'),
[Produce Unit]('worker', 'east'),
[Produce Unit]('worker', 'south'),
[Produce Unit]('ranged', 'east'),
[Produce Unit]('ranged', 'south'),
[Deploy Unit]('ranged', (3, 5)),
[Deploy Unit]('worker', (3, 4)),
[Deploy Unit]('worker', (7, 5)),
END of TASK
"""

counter_coacAI_600_to_inf = """
START of TASK
[Produce Unit]('worker', 'east'),
[Produce Unit]('worker', 'south'),
[Produce Unit]('worker', 'east'),
[Produce Unit]('worker', 'south'),
[Deploy Unit]('worker', (2, 3)),
[Deploy Unit]('worker', (2, 5)),
[Produce Unit]('ranged', 'east'),
[Produce Unit]('ranged', 'south'),
[Produce Unit]('light', 'south'),
[Produce Unit]('ranged', 'east'),
[Produce Unit]('ranged', 'south'),
[Deploy Unit]('ranged', (4, 5)),
[Deploy Unit]('ranged', (5, 6)),
[Deploy Unit]('ranged', (5, 4)),
[Deploy Unit]('ranged', (7, 4)),
END of TASK
"""

counter_coacAI = [
    counter_coacAI_0_to_200,
    counter_coacAI_200_to_400,
    counter_coacAI_400_to_600,
    counter_coacAI_600_to_inf,
    counter_coacAI_600_to_inf,
]
