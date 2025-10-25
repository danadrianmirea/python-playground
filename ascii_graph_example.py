from ascii_graph import Pyasciigraph

test = [('long_label', 423), ('sl', 1234), ('line3', 531),
    ('line4', 200), ('line5', 834)]

graph = Pyasciigraph()
for line in  graph.graph('test print', test):
    print(line)