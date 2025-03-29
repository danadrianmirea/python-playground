import ast
import networkx as nx
import sys
import traceback
# Set the backend before importing pyplot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def create_cfg(code):
    try:
        print("Parsing code into AST...")
        # Parse the code into an AST
        tree = ast.parse(code)
        print("AST parsed successfully")
        
        # Create a directed graph
        G = nx.DiGraph()
        print("Created directed graph")
        
        # Keep track of the current node ID
        node_id = 0
        
        def add_node(label, parent=None):
            nonlocal node_id
            node_id += 1
            node = f"{label}-{node_id}"
            G.add_node(node)
            if parent:
                G.add_edge(parent, node)
            return node
        
        def visit_node(node, parent=None):
            try:
                if isinstance(node, ast.FunctionDef):
                    # Add function entry node
                    func_node = add_node(f"Function: {node.name}", parent)
                    # Process function body
                    for stmt in node.body:
                        visit_node(stmt, func_node)
                        
                elif isinstance(node, ast.If):
                    # Add if condition node
                    if_node = add_node("If", parent)
                    # Visit the test condition
                    visit_node(node.test, if_node)
                    # Add true branch
                    true_node = add_node("True Branch", if_node)
                    for stmt in node.body:
                        visit_node(stmt, true_node)
                    # Add false branch
                    if node.orelse:
                        false_node = add_node("False Branch", if_node)
                        for stmt in node.orelse:
                            visit_node(stmt, false_node)
                        
                elif isinstance(node, ast.Return):
                    # Add return node
                    return_node = add_node("Return", parent)
                    if node.value:
                        if isinstance(node.value, ast.Constant):
                            add_node(f"Value: {node.value.value}", return_node)
                        elif isinstance(node.value, ast.Name):
                            add_node(f"Variable: {node.value.id}", return_node)
                            
                elif isinstance(node, ast.Compare):
                    # Add comparison node
                    comp_node = add_node("Compare", parent)
                    # Add left side of comparison
                    if isinstance(node.left, ast.Name):
                        add_node(f"Variable: {node.left.id}", comp_node)
                    elif isinstance(node.left, ast.Constant):
                        add_node(f"Value: {node.left.value}", comp_node)
                    # Add operator
                    for op in node.ops:
                        add_node(f"Operator: {type(op).__name__}", comp_node)
                    # Add right side of comparison
                    for comp in node.comparators:
                        if isinstance(comp, ast.Constant):
                            add_node(f"Value: {comp.value}", comp_node)
                        elif isinstance(comp, ast.Name):
                            add_node(f"Variable: {comp.id}", comp_node)
                
                elif isinstance(node, ast.Assign):
                    # Add assignment node
                    assign_node = add_node("Assign", parent)
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            add_node(f"Target: {target.id}", assign_node)
                    if isinstance(node.value, ast.Constant):
                        add_node(f"Value: {node.value.value}", assign_node)
                    elif isinstance(node.value, ast.Name):
                        add_node(f"Value: {node.value.id}", assign_node)
                        
                elif isinstance(node, ast.Module):
                    # Process all statements in the module
                    for stmt in node.body:
                        visit_node(stmt)
            except Exception as e:
                print(f"Error in visit_node: {str(e)}")
                print(f"Node type: {type(node).__name__}")
                traceback.print_exc()
                raise
        
        print("Starting to visit nodes...")
        # Start visiting from the root
        visit_node(tree)
        print(f"Graph created successfully with {len(G.nodes)} nodes and {len(G.edges)} edges")
        return G
    except SyntaxError as e:
        print(f"Syntax error in the code: {e}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"Error creating control flow graph: {e}")
        traceback.print_exc()
        sys.exit(1)

def visualize_cfg(G):
    try:
        print("Creating visualization...")
        # Create a new figure with a larger size
        plt.figure(figsize=(15, 10))
        
        # Use a hierarchical layout for better visualization
        print("Computing layout...")
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        print("Drawing graph...")
        # Draw the graph with improved styling
        nx.draw(G, pos,
                with_labels=True,
                node_color='lightblue',
                node_size=3000,
                font_size=10,
                font_weight='bold',
                arrows=True,
                edge_color='gray',
                arrowsize=20,
                edgecolors='black')
        
        # Add a title
        plt.title("Control Flow Graph", pad=20, size=16)
        
        # Force the plot to render
        plt.draw()
        
        print("Showing plot...")
        # Block until the window is closed
        plt.show(block=True)
        
        # Just in case the window was closed
        input("Press Enter to continue...")
        
    except Exception as e:
        print(f"Error visualizing the graph: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        print("Starting program...")
        # Sample Python code
        code = """def test(a):
    if a > 0:
        return 'Positive'
    else:
        return 'Non-positive'
"""
        
        # Create and visualize the control flow graph
        G = create_cfg(code)
        visualize_cfg(G)
    except Exception as e:
        print(f"Main program error: {e}")
        traceback.print_exc()
        sys.exit(1)