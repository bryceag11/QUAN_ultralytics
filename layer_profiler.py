import torch
import numpy as np
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import gc
from contextlib import contextmanager
import sys

sys.path.append('.') 
from ultralytics import YOLO


class LayerProfiler:
    """
    A more detailed profiler focused on specific layer types in the model backbone
    """
    
    def __init__(self, 
                 q_model_path,
                 regular_model_path=None,
                 imgsz=640, 
                 num_warmup_runs=10, 
                 num_timed_runs=50,
                 focus_layer_types=None,
                 device=None):
        """
        Initialize the LayerProfiler.
        
        Args:
            q_model_path (str): Path to quaternion model file or config
            regular_model_path (str, optional): Path to regular model file or config
            imgsz (int): Image size for inference
            num_warmup_runs (int): Number of warmup runs before timing
            num_timed_runs (int): Number of timed runs for averaging
            focus_layer_types (list): List of layer types to focus on (e.g., ['Conv', 'QConv'])
            device (str, optional): Device to run on ('cpu', '0', etc.)
        """
        self.q_model_path = q_model_path
        self.regular_model_path = regular_model_path
        self.imgsz = imgsz
        self.num_warmup_runs = num_warmup_runs
        self.num_timed_runs = num_timed_runs
        self.focus_layer_types = focus_layer_types or ['Conv', 'QConv', 'C3', 'C2f', 'SPPF', 'QSPPF', 'Detect']
        self.device = device or ('0' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        print(f"Loading quaternion model from {q_model_path}...")
        self.q_model = YOLO(q_model_path)
        
        if regular_model_path:
            print(f"Loading regular model from {regular_model_path}...")
            self.regular_model = YOLO(regular_model_path)
        else:
            self.regular_model = None
        
        # Create input data
        self.create_input_data()
        
        # Results storage
        self.layer_details = {
            'quaternion': {},
            'regular': {}
        }
        
        self.timing_results = {
            'quaternion': {},
            'regular': {}
        }
    
    def create_input_data(self):
        """Create synthetic input data for inference."""
        # RGB input for both models
        self.input_data = torch.randint(0, 255, (1, 3, self.imgsz, self.imgsz), 
                                        dtype=torch.uint8, 
                                        device=self.device)
    
    def get_model_structure(self, model, model_type):
        """
        Extract detailed structure of the model with layer types and parameter counts.
        
        Args:
            model: YOLO model
            model_type: 'quaternion' or 'regular'
            
        Returns:
            dict: Structured representation of model layers
        """
        result = {}
        
        # Access model layers
        if hasattr(model.model, 'model'):
            layers = model.model.model
        else:
            layers = model.model
        
        # Process each layer
        for i, layer in enumerate(layers):
            layer_name = f"{layer.type}_{layer.i}"
            
            # Get parameter count
            param_count = sum(p.numel() for p in layer.parameters())
            
            # Get input/output shapes if possible
            if hasattr(layer, 'conv') and hasattr(layer.conv, 'in_channels'):
                in_channels = layer.conv.in_channels
                out_channels = layer.conv.out_channels
            else:
                in_channels = out_channels = None
            
            # Store layer details
            result[layer_name] = {
                'type': layer.type,
                'index': layer.i,
                'from': layer.f,
                'params': param_count,
                'in_channels': in_channels,
                'out_channels': out_channels
            }
        
        # Store in class attribute
        self.layer_details[model_type] = result
        return result
    
    def profile_backbone_layers(self, model, model_type):
        """
        Profile specific layers in the model backbone to isolate performance differences.
        
        Args:
            model: YOLO model
            model_type: 'quaternion' or 'regular'
            
        Returns:
            dict: Performance metrics for backbone layers
        """
        # Get model structure first
        structure = self.get_model_structure(model, model_type)
        
        # Access model layers
        if hasattr(model.model, 'model'):
            layers = model.model.model
        else:
            layers = model.model
        
        # Prepare hooks
        hooks = []
        layer_times = {}
        
        def create_timing_hook(layer_name):
            """Create a timing hook for a specific layer."""
            times = []
            
            def hook(module, input, output):
                # Start timing
                start = time.perf_counter()
                # Call original forward
                result = module.forward_original(input[0])
                # End timing
                end = time.perf_counter()
                # Record time in ms
                times.append((end - start) * 1000)
                return result
            
            layer_times[layer_name] = times
            return hook
        
        # Register hooks for focused layer types
        for i, layer in enumerate(layers):
            layer_name = f"{layer.type}_{layer.i}"
            
            # Only profile layers of specified types
            if any(type_name in layer.type for type_name in self.focus_layer_types):
                # Store original forward method
                layer.forward_original = layer.forward
                # Create and register hook
                hook_fn = create_timing_hook(layer_name)
                hook = layer.register_forward_hook(hook_fn)
                hooks.append((layer, hook))
        
        # Warmup runs
        print(f"Warming up {model_type} model with {self.num_warmup_runs} runs...")
        for _ in range(self.num_warmup_runs):
            with torch.no_grad():
                model(self.input_data, verbose=False)
        
        # Timed runs
        print(f"Profiling {model_type} model with {self.num_timed_runs} runs...")
        for _ in tqdm(range(self.num_timed_runs)):
            # Clear cache before each run
            if self.device != 'cpu':
                torch.cuda.empty_cache()
            gc.collect()
            
            # Run inference
            with torch.no_grad():
                model(self.input_data, verbose=False)
        
        # Process results
        layer_avg_times = {}
        for layer_name, times in layer_times.items():
            if len(times) >= 2:  # Skip first run as warmup within timing loop
                avg_time = np.mean(times[1:])
                std_time = np.std(times[1:])
                layer_avg_times[layer_name] = {
                    'avg_time': avg_time,
                    'std_time': std_time
                }
        
        # Remove hooks and restore original forward methods
        for layer, hook in hooks:
            hook.remove()
            delattr(layer, 'forward_original')
        
        # Save results
        self.timing_results[model_type] = layer_avg_times
        return layer_avg_times
    
    def profile(self):
        """Run profiling on both models, focusing on backbone layers."""
        print("\n--- Starting backbone layer profiling ---")
        
        # Profile quaternion model
        q_backbone_results = self.profile_backbone_layers(self.q_model, 'quaternion')
        
        # Profile regular model if available
        if self.regular_model:
            reg_backbone_results = self.profile_backbone_layers(self.regular_model, 'regular')
        
        return {
            'quaternion': q_backbone_results,
            'regular': reg_backbone_results if self.regular_model else None,
            'layer_details': self.layer_details
        }
    
    def analyze_layer_types(self):
        """
        Analyze performance by layer type instead of individual layers.
        This helps identify which types of layers benefit most from quaternion implementation.
        """
        if not self.regular_model:
            print("Need both quaternion and regular models for layer type analysis.")
            return {}
        
        # Group timing by layer type
        q_by_type = {}
        reg_by_type = {}
        
        # Process quaternion model results
        for layer_name, timing in self.timing_results['quaternion'].items():
            layer_type = layer_name.split('_')[0]
            if layer_type not in q_by_type:
                q_by_type[layer_type] = []
            q_by_type[layer_type].append(timing['avg_time'])
        
        # Process regular model results
        for layer_name, timing in self.timing_results['regular'].items():
            layer_type = layer_name.split('_')[0]
            if layer_type not in reg_by_type:
                reg_by_type[layer_type] = []
            reg_by_type[layer_type].append(timing['avg_time'])
        
        # Calculate averages by type
        type_comparison = {}
        for layer_type in set(list(q_by_type.keys()) + list(reg_by_type.keys())):
            q_times = q_by_type.get(layer_type, [])
            reg_times = reg_by_type.get(layer_type, [])
            
            if q_times and reg_times:
                q_avg = np.mean(q_times)
                reg_avg = np.mean(reg_times)
                ratio = q_avg / reg_avg
                
                type_comparison[layer_type] = {
                    'quaternion_avg': q_avg,
                    'regular_avg': reg_avg,
                    'ratio': ratio,
                    'q_count': len(q_times),
                    'reg_count': len(reg_times)
                }
        
        return type_comparison
    
    def print_summary(self):
        """Print a detailed summary of the profiling results with focus on backbone layers."""
        # Print quaternion model results
        print("\n--- Quaternion Model Layer Timing ---")
        if self.timing_results['quaternion']:
            sorted_layers = sorted(
                self.timing_results['quaternion'].items(),
                key=lambda x: x[1]['avg_time'],
                reverse=True
            )
            
            print(f"{'Layer':<20} {'Time (ms)':<12} {'Std Dev':<12} {'Parameters':<12}")
            print("-" * 60)
            
            for layer_name, timing in sorted_layers[:10]:  # Top 10 layers
                layer_details = self.layer_details['quaternion'].get(layer_name, {})
                params = layer_details.get('params', 0)
                
                print(f"{layer_name:<20} {timing['avg_time']:<12.3f} {timing['std_time']:<12.3f} {params:<12,}")
        
        # Print regular model results if available
        if self.regular_model and self.timing_results['regular']:
            print("\n--- Regular Model Layer Timing ---")
            sorted_layers = sorted(
                self.timing_results['regular'].items(),
                key=lambda x: x[1]['avg_time'],
                reverse=True
            )
            
            print(f"{'Layer':<20} {'Time (ms)':<12} {'Std Dev':<12} {'Parameters':<12}")
            print("-" * 60)
            
            for layer_name, timing in sorted_layers[:10]:  # Top 10 layers
                layer_details = self.layer_details['regular'].get(layer_name, {})
                params = layer_details.get('params', 0)
                
                print(f"{layer_name:<20} {timing['avg_time']:<12.3f} {timing['std_time']:<12.3f} {params:<12,}")
        
        # If both models are available, print layer type comparison
        if self.regular_model:
            type_analysis = self.analyze_layer_types()
            if type_analysis:
                print("\n--- Layer Type Comparison (Quaternion vs Regular) ---")
                print(f"{'Layer Type':<15} {'Q Time (ms)':<12} {'R Time (ms)':<12} {'Ratio (Q/R)':<15} {'Q Layers':<10} {'R Layers':<10}")
                print("-" * 80)
                
                # Sort by ratio to highlight biggest differences
                sorted_types = sorted(
                    type_analysis.items(),
                    key=lambda x: x[1]['ratio'],
                    reverse=True
                )
                
                for layer_type, data in sorted_types:
                    print(f"{layer_type:<15} {data['quaternion_avg']:<12.3f} {data['regular_avg']:<12.3f} {data['ratio']:<15.3f} {data['q_count']:<10} {data['reg_count']:<10}")
    
    def plot_layer_comparison(self, save_path=None):
        """
        Create visualizations comparing layer performance between models.
        
        Args:
            save_path (str, optional): Path prefix for saving plots
        """
        if not self.regular_model:
            print("Need both quaternion and regular models for comparison plots.")
            return
        
        # 1. Layer Type Comparison Plot
        type_analysis = self.analyze_layer_types()
        if type_analysis:
            # Prepare data
            types = list(type_analysis.keys())
            q_times = [data['quaternion_avg'] for data in type_analysis.values()]
            r_times = [data['regular_avg'] for data in type_analysis.values()]
            ratios = [data['ratio'] for data in type_analysis.values()]
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # First subplot: Times comparison
            x = np.arange(len(types))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, q_times, width, label='Quaternion')
            bars2 = ax1.bar(x + width/2, r_times, width, label='Regular')
            
            ax1.set_ylabel('Time (ms)')
            ax1.set_title('Layer Type Timing Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(types, rotation=45, ha='right')
            ax1.legend()
            
            # Second subplot: Ratio
            color_map = ['green' if r < 1 else 'red' for r in ratios]
            bars3 = ax2.bar(types, ratios, color=color_map)
            ax2.axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
            
            ax2.set_ylabel('Ratio (Quaternion/Regular)')
            ax2.set_title('Relative Performance')
            ax2.set_xticklabels(types, rotation=45, ha='right')
            
            # Add labels on top of bars
            for bar, ratio in zip(bars3, ratios):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{ratio:.2f}x', ha='center', va='bottom')
            
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}_layer_types.png", dpi=300, bbox_inches='tight')
            
        # 2. Individual Backbone Layers Plot
        # Find common layers between the two models
        q_layers = set(self.timing_results['quaternion'].keys())
        r_layers = set(self.timing_results['regular'].keys())
        common_layers = q_layers.intersection(r_layers)
        
        if common_layers:
            # Prepare data for common layers
            layer_names = []
            q_times = []
            r_times = []
            speedups = []
            
            for layer in common_layers:
                layer_names.append(layer)
                q_time = self.timing_results['quaternion'][layer]['avg_time']
                r_time = self.timing_results['regular'][layer]['avg_time']
                q_times.append(q_time)
                r_times.append(r_time)
                speedups.append(q_time / r_time)
            
            # Sort by regular model time to highlight most important layers
            sorted_indices = np.argsort(r_times)[::-1][:10]  # Top 10 by time
            layer_names = [layer_names[i] for i in sorted_indices]
            q_times = [q_times[i] for i in sorted_indices]
            r_times = [r_times[i] for i in sorted_indices]
            speedups = [speedups[i] for i in sorted_indices]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(14, 8))
            
            x = np.arange(len(layer_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, q_times, width, label='Quaternion')
            bars2 = ax.bar(x + width/2, r_times, width, label='Regular')
            
            ax.set_ylabel('Time (ms)')
            ax.set_title('Common Layer Timing Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(layer_names, rotation=45, ha='right')
            ax.legend()
            
            # Add speedup as text above bars
            for i, (q, r, s) in enumerate(zip(q_times, r_times, speedups)):
                ax.text(i, max(q, r) + 0.1, f'{s:.2f}x', ha='center')
            
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}_common_layers.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_results(self, output_path=None):
        """
        Export detailed profiling results to CSV files.
        
        Args:
            output_path (str, optional): Path prefix for output files
        """
        if output_path is None:
            output_path = "backbone_profiling"
        
        # 1. Export layer timing results
        q_data = []
        for layer_name, timing in self.timing_results['quaternion'].items():
            layer_details = self.layer_details['quaternion'].get(layer_name, {})
            q_data.append({
                'model_type': 'quaternion',
                'layer_name': layer_name,
                'layer_type': layer_name.split('_')[0],
                'time_ms': timing['avg_time'],
                'std_ms': timing['std_time'],
                'params': layer_details.get('params', 0),
                'in_channels': layer_details.get('in_channels'),
                'out_channels': layer_details.get('out_channels')
            })
        
        r_data = []
        if self.regular_model:
            for layer_name, timing in self.timing_results['regular'].items():
                layer_details = self.layer_details['regular'].get(layer_name, {})
                r_data.append({
                    'model_type': 'regular',
                    'layer_name': layer_name,
                    'layer_type': layer_name.split('_')[0],
                    'time_ms': timing['avg_time'],
                    'std_ms': timing['std_time'],
                    'params': layer_details.get('params', 0),
                    'in_channels': layer_details.get('in_channels'),
                    'out_channels': layer_details.get('out_channels')
                })
        
        # Combine and save
        all_data = q_data + r_data
        df = pd.DataFrame(all_data)
        layer_output = f"{output_path}_layer_details.csv"
        df.to_csv(layer_output, index=False)
        print(f"Layer details exported to {layer_output}")
        
        # 2. Export layer type analysis
        if self.regular_model:
            type_analysis = self.analyze_layer_types()
            if type_analysis:
                type_data = []
                for layer_type, data in type_analysis.items():
                    type_data.append({
                        'layer_type': layer_type,
                        'quaternion_time_ms': data['quaternion_avg'],
                        'regular_time_ms': data['regular_avg'],
                        'ratio': data['ratio'],
                        'quaternion_count': data['q_count'],
                        'regular_count': data['reg_count'],
                        'speedup': 1/data['ratio'] if data['ratio'] > 1 else data['ratio']
                    })
                
                df = pd.DataFrame(type_data)
                type_output = f"{output_path}_type_analysis.csv"
                df.to_csv(type_output, index=False)
                print(f"Layer type analysis exported to {type_output}")


def main():
    parser = argparse.ArgumentParser(description='Profile Quaternion vs Regular YOLO Model Backbone Layers')
    parser.add_argument('--q_model', type=str, required=True, help='Path to quaternion model')
    parser.add_argument('--reg_model', type=str, default=None, help='Path to regular model for comparison')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for inference')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup runs')
    parser.add_argument('--runs', type=int, default=50, help='Number of timed runs')
    parser.add_argument('--device', type=str, default=None, help='Device to run on (cpu, 0, 1, etc.)')
    parser.add_argument('--output', type=str, default='backbone_profiling', help='Output filename prefix')
    parser.add_argument('--layer_types', type=str, nargs='+', 
                      default=['Conv', 'QConv', 'C3', 'C2f', 'SPPF', 'QSPPF', 'Detect'],
                      help='Layer types to focus on')
    args = parser.parse_args()
    
    profiler = LayerProfiler(
        q_model_path=args.q_model,
        regular_model_path=args.reg_model,
        imgsz=args.imgsz,
        num_warmup_runs=args.warmup,
        num_timed_runs=args.runs,
        focus_layer_types=args.layer_types,
        device=args.device
    )
    
    results = profiler.profile()
    profiler.print_summary()
    profiler.plot_layer_comparison(save_path=args.output)
    profiler.export_results(output_path=args.output)


if __name__ == "__main__":
    main()
