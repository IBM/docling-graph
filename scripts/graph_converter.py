"""
Handles the conversion of Pydantic models to a NetworkX graph structure.

This version includes:
- Content-based entity resolution for deduplication.
- A method to export the graph to a backend-agnostic JSON format.
"""
from pydantic import BaseModel
import networkx as nx
import hashlib

class GraphConverter:
    """
    Converts Pydantic models into a NetworkX graph with entity deduplication.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self._visited_ids = set()

    def _get_node_id(self, model_instance: BaseModel) -> str:
        """
        Creates a deterministic, content-based ID for a node to enable deduplication.
        """
        node_label = model_instance.__class__.__name__
        
        # Define key fields for creating a unique signature for each entity type
        id_fields = {
            "Person": ["name"],
            "Organization": ["name"],
            "Address": ["street", "city", "postal_code"],
            "Invoice": ["bill_no"],
        }.get(node_label, [])

        # For types like LineItem or others without a clear key, we hash all properties
        if not id_fields:
            # Fallback to hashing all model data for a stable ID
            model_string = str(model_instance.model_dump())
        else:
            # Create a stable string from the key fields
            model_string = "|".join(
                str(getattr(model_instance, f, "")).lower() for f in id_fields
            )
            
        # Use a SHA256 hash to create a unique and consistent ID
        hash_id = hashlib.sha256(model_string.encode()).hexdigest()
        return f"{node_label}_{hash_id[:12]}" # Use a truncated hash for brevity

    def _process_model(self, model_instance: BaseModel):
        if not isinstance(model_instance, BaseModel):
            return

        current_node_id = self._get_node_id(model_instance)
        
        # Add node only if it's the first time we see this unique ID
        if current_node_id not in self.graph:
            node_properties = {
                k: v for k, v in model_instance.model_dump().items()
                if not (model_instance.model_fields[k].json_schema_extra and 'edge_label' in model_instance.model_fields[k].json_schema_extra)
            }
            self.graph.add_node(
                current_node_id,
                label=model_instance.__class__.__name__,
                **node_properties
            )

        # Recursion logic remains the same, but now uses the new deterministic IDs
        for field_name, field_info in model_instance.model_fields.items():
            if field_info.json_schema_extra and 'edge_label' in field_info.json_schema_extra:
                edge_label = field_info.json_schema_extra['edge_label']
                child_objects = getattr(model_instance, field_name)
                
                if not isinstance(child_objects, list):
                    child_objects = [child_objects]

                for item in child_objects:
                    if isinstance(item, BaseModel):
                        child_node_id = self._get_node_id(item)
                        # Ensure the child node is processed
                        self._process_model(item)
                        # Add the edge
                        self.graph.add_edge(current_node_id, child_node_id, label=edge_label)

    def pydantic_to_graph(self, root_model: BaseModel) -> nx.DiGraph:
        self._process_model(root_model)
        return self.graph

    def to_json_serializable(self) -> dict:
        """
        Exports the graph to a backend-agnostic, JSON-friendly format.

        Returns:
            A dictionary with two keys: 'nodes' and 'edges', containing lists
            of dictionaries representing the graph's data.
        """
        graph_data = nx.node_link_data(self.graph)
        
        # Reformat for clarity and common graph DB import formats
        nodes = [{"id": n["id"], "type": n.get("label", ""), "properties": {k:v for k,v in n.items() if k not in ["id", "label"]}} for n in graph_data["nodes"]]
        edges = [{"source": e["source"], "target": e["target"], "label": e.get("label", "")} for e in graph_data["links"]]
        
        return {"nodes": nodes, "edges": edges}

