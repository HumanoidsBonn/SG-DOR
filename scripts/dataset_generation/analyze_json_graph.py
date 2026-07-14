import json
import argparse
from collections import Counter

def analyze_json(file_path):
    print(f"Reading {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)

    all_nodes = []
    for frame in data:
        nodes = frame.get("nodes", [])
        all_nodes.extend(nodes)

    print(f"Total nodes found: {len(all_nodes)}")

    # Summary of Types
    types = [node.get("type", "N/A") for node in all_nodes]
    type_counts = Counter(types)
    print("\n--- Semantic Types Summary ---")
    for t, count in type_counts.items():
        print(f"  {t:12}: {count}")

    # Summary of Prototypes
    prototypes = [node.get("prototype", "N/A") for node in all_nodes]
    proto_counts = Counter(prototypes)
    print("\n--- Prototypes Summary ---")
    for p, count in proto_counts.most_common(20):
        print(f"  {p:15}: {count}")
    if len(proto_counts) > 20:
        print(f"  ... and {len(proto_counts) - 20} more")

    # Detailed listing of first few nodes for inspection
    print("\n--- First 20 Nodes (ID, Type, Prototype) ---")
    for node in all_nodes[:20]:
        print(f"  ID: {node['id']:30} | Type: {node['type']:10} | Proto: {node['prototype']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="Path to the scene graph JSON file")
    args = parser.parse_args()
    analyze_json(args.file)
