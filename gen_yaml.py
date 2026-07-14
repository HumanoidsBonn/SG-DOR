import yaml
import ast

def extract_args():
    with open('scripts/train_oracle.py', 'r') as f:
        src = f.read()
    
    tree = ast.parse(src)
    config = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            arg_name = None
            default_val = None
            action = None
            
            for arg in node.args:
                if isinstance(arg, ast.Constant) and arg.value.startswith('--'):
                    arg_name = arg.value.lstrip('-').replace('-', '_')
                    # Prefer the first long arg
                    break
            
            if not arg_name: continue
            
            for kw in node.keywords:
                if kw.arg == 'default':
                    if isinstance(kw.value, ast.Constant):
                        default_val = kw.value.value
                    elif isinstance(kw.value, ast.Name) and kw.value.id == 'None':
                        default_val = None
                elif kw.arg == 'action':
                    if isinstance(kw.value, ast.Constant):
                        action = kw.value.value
                        
            if action == 'store_true':
                if default_val is None: default_val = False
            elif action == 'store_false':
                if default_val is None: default_val = True
                
            config[arg_name] = default_val
            
    return config

config = extract_args()
with open('oracle_config.yaml', 'w') as f:
    yaml.dump(config, f, sort_keys=False)
print("oracle_config.yaml generated successfully!")
