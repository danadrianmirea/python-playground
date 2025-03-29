import ast
import networkx as nx
import matplotlib.pyplot as plt
import sys

def create_cfg(code):
    try:
        # Parse the code into an AST
        tree = ast.parse(code)
        
        # Create a directed graph
        G = nx.DiGraph()
        
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
            if isinstance(node, ast.FunctionDef):
                # Add function entry node
                func_node = add_node(f"Function: {node.name}", parent)
                # Process function body
                for stmt in node.body:
                    visit_node(stmt, func_node)
                    
            elif isinstance(node, ast.If):
                # Add if condition node
                if_node = add_node("If", parent)
                # Add test condition
                test_node = add_node("Test", if_node)
                # Add true branch
                for stmt in node.body:
                    visit_node(stmt, if_node)
                # Add false branch
                for stmt in node.orelse:
                    visit_node(stmt, if_node)
                    
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
        
        # Start visiting from the root
        visit_node(tree)
        return G
    except SyntaxError as e:
        print(f"Syntax error in the code: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error creating control flow graph: {e}")
        sys.exit(1)

def visualize_cfg(G):
    try:
        # Create a new figure with a larger size
        plt.figure(figsize=(15, 10))
        
        # Use a hierarchical layout for better visualization
        pos = nx.spring_layout(G, k=2, iterations=50)
        
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
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Show the plot
        plt.show()
    except Exception as e:
        print(f"Error visualizing the graph: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Sample Python code
    code = """
    def test(a):
        if a > 0:
            return 'Positive'
        else:
            return 'Non-positive'
    """
    
    # Create and visualize the control flow graph
    G = create_cfg(code)
    visualize_cfg(G)