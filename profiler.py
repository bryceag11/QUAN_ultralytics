import torch
import torch.nn as nn
import time
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import contextmanager
import gc
import sys
from tqdm import tqdm

sys.path.append('.')  # Add current directory to path
from ultralytics import YOLO


@contextmanager
def timer_context():
    """Context manager for timing code execution."""
    start = time.time()
    yield
    end = time.time()
    elapsed_time = (end - start) * 1000  # milliseconds
    return elapsed_time


class ModelProfiler:
    """
    A profiler for comparing quaternion and regular YOLO models.
    Focuses on layer-by-layer timing and memory usage without ONNX exports.
    """
    
    def __init__(self, 
                 q_model_path,
                 regular_model_path=None,
                 imgsz=640, 
                 num_warmup_runs=10, 
                 num_timed_runs=100,
                 device=None, 
                 input_data=None):
        """
        Initialize the ModelProfiler.
        
        Args:
            q_model_path (str): Path to quaternion model file or config
            regular_model_path (str, optional): Path to regular model file or config
            imgsz (int): Image size for inference
            num_warmup_runs (int): Number of warmup runs before timing
            num_timed_runs (int): Number of timed runs for averaging
            device (str, optional): Device to run on ('cpu', '0', etc.)
        """
        self.q_model_path = q_model_path
        self.regular_model_path = regular_model_path
        self.imgsz = imgsz
        self.num_warmup_runs = num_warmup_runs
        self.num_timed_runs = num_timed_runs
        self.device = device or ('0' if torch.cuda.is_available() else 'cpu')
        
        # Model loading
        print(f"Loading quaternion model from {q_model_path}...")
        self.q_model = YOLO(q_model_path)
        
        if regular_model_path:
            print(f"Loading regular model from {regular_model_path}...")
            self.regular_model = YOLO(regular_model_path)
        else:
            self.regular_model = None
        
        # Create synthetic input data
        # self.create_input_data()
        
        # Results storage
        self.results = {
            'quaternion': {
                'total_time': 0,
                'layer_times': defaultdict(list),
                'layer_memory': defaultdict(list),
                'params': 0,
                'flops': 0
            }
        }
        
        if self.regular_model:
            self.results['regular'] = {
                'total_time': 0,
                'layer_times': defaultdict(list),
                'layer_memory': defaultdict(list),
                'params': 0,
                'flops': 0
            }
    
    def create_input_data(self):
        """Create synthetic input data for inference."""
        self.input_data = (
            torch.randint(0, 255, (1, 3, self.imgsz, self.imgsz), dtype=torch.uint8)
            .float()
            .div(255.0)
            .to(self.device)
        )
        
    def measure_layer_performance(self, model, model_type='quaternion'):
        """
        Measure performance of each layer in the model.
        
        Args:
            model: The YOLO model to profile
            model_type (str): Either 'quaternion' or 'regular'
            
        Returns:
            dict: Performance metrics
        """
        results = self.results[model_type]
        model_instance = model.model
        
        # Get access to the model
        if hasattr(model_instance, 'model'):
            layers = model_instance.model
        else:
            layers = model_instance
        
        # Register hooks for all layers
        hooks = []
        layer_times = defaultdict(list)
        layer_memory = defaultdict(list)
        
        def forward_hook(name):
            def hook(module, input, output):
                start = time.perf_counter()
                result = module.forward_original(input[0])
                end = time.perf_counter()
                layer_times[name].append((end - start) * 1000)  # ms
                
                # Memory usage if GPU
                if self.device != 'cpu':
                    mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                    layer_memory[name].append(mem)
                
                return result
            return hook
        
        # Add hooks to each layer
        for i, layer in enumerate(layers):
            layer_name = f"{layer.type}_{layer.i}"
            # Store original forward method
            layer.forward_original = layer.forward
            # Create and register a new forward method
            hook_fn = forward_hook(layer_name)
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        # Warmup runs
        for _ in range(self.num_warmup_runs):
            with torch.no_grad():
                for batch in self.input_data:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    model(x.to(self.device), verbose=False)
                    break


        
        # Timed runs
        total_times = []
        for _ in tqdm(range(self.num_timed_runs), desc=f"Profiling {model_type} model"):
            # Clear cache before each run
            if self.device != 'cpu':
                torch.cuda.empty_cache()
            gc.collect()
            
            # Time full model
            start = time.perf_counter()
            with torch.no_grad():
                model(self.input_data, verbose=False)
            end = time.perf_counter()
            total_times.append((end - start) * 1000)  # ms
        
        # Remove hooks and restore original forward methods
        for i, layer in enumerate(layers):
            hooks[i].remove()
            delattr(layer, 'forward_original')
        
        # Process results
        results['total_time'] = np.mean(total_times)
        results['total_std'] = np.std(total_times)
        
        for name in layer_times:
            results['layer_times'][name] = np.mean(layer_times[name])
            if name in layer_memory:
                results['layer_memory'][name] = np.mean(layer_memory[name])
        
        # Calculate parameters
        results['params'] = sum([p.numel() for p in model.parameters() if p.requires_grad])
        
        return results
    
    def profile(self):
        """Run full profiling on both models."""
        print("\n--- Starting profiling ---")
        
        # Profile quaternion model
        q_results = self.measure_layer_performance(self.q_model, 'quaternion')
        
        # Profile regular model if available
        if self.regular_model:
            reg_results = self.measure_layer_performance(self.regular_model, 'regular')
        
        return self.results
    
    def compare_results(self):
        """Compare results between quaternion and regular models."""
        if not self.regular_model:
            print("No regular model provided for comparison.")
            return self.results['quaternion']
        
        q_results = self.results['quaternion']
        reg_results = self.results['regular']
        
        comparison = {
            'total_time_ratio': q_results['total_time'] / reg_results['total_time'],
            'params_ratio': q_results['params'] / reg_results['params'],
            'layer_time_ratios': {}
        }
        
        # Find common layers
        q_layers = set(q_results['layer_times'].keys())
        reg_layers = set(reg_results['layer_times'].keys())
        common_layers = q_layers.intersection(reg_layers)
        
        for layer in common_layers:
            q_time = q_results['layer_times'][layer]
            reg_time = reg_results['layer_times'][layer]
            comparison['layer_time_ratios'][layer] = q_time / reg_time
        
        return comparison
    
    def print_summary(self):
        """Print a summary of profiling results."""
        q_results = self.results['quaternion']
        
        print("\n--- Quaternion Model Summary ---")
        print(f"Total inference time: {q_results['total_time']:.2f} ± {q_results.get('total_std', 0):.2f} ms")
        print(f"Parameters: {q_results['params']:,}")
        
        print("\nTop 5 slowest layers:")
        layers_by_time = sorted(q_results['layer_times'].items(), key=lambda x: x[1], reverse=True)
        for name, time in layers_by_time[:5]:
            print(f"  {name}: {time:.2f} ms")
        
        if self.regular_model:
            reg_results = self.results['regular']
            comparison = self.compare_results()
            
            print("\n--- Regular Model Summary ---")
            print(f"Total inference time: {reg_results['total_time']:.2f} ± {reg_results.get('total_std', 0):.2f} ms")
            print(f"Parameters: {reg_results['params']:,}")
            
            print("\n--- Comparison ---")
            print(f"Speed ratio (Q/Reg): {comparison['total_time_ratio']:.2f}x")
            print(f"Parameter ratio (Q/Reg): {comparison['params_ratio']:.2f}x")
            
            if comparison['total_time_ratio'] < 1:
                print(f"Quaternion model is {1/comparison['total_time_ratio']:.2f}x faster")
            else:
                print(f"Regular model is {comparison['total_time_ratio']:.2f}x faster")
    
    def plot_results(self, save_path=None):
        """
        Plot comparison results.
        
        Args:
            save_path (str, optional): Path to save plots
        """
        if not self.regular_model:
            print("No regular model provided for visualization.")
            return
        
        # 1. Overall time comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        models = ['Quaternion', 'Regular']
        times = [
            self.results['quaternion']['total_time'],
            self.results['regular']['total_time']
        ]
        bars = ax.bar(models, times, color=['blue', 'orange'])
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('Model Inference Time Comparison')
        
        # Add time values on top of bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{time:.2f} ms', ha='center', va='bottom')
            
        if save_path:
            plt.savefig(f"{save_path}_time_comparison.png", dpi=300, bbox_inches='tight')
        
        # 2. Layer-by-layer comparison
        q_layers = self.results['quaternion']['layer_times']
        reg_layers = self.results['regular']['layer_times']
        
        # Find common layers
        common_layers = set(q_layers.keys()).intersection(set(reg_layers.keys()))
        
        if common_layers:
            # Sort layers by regular model time
            sorted_layers = sorted(common_layers, key=lambda x: reg_layers[x], reverse=True)
            top_n = min(10, len(sorted_layers))  # Show top 10 layers or fewer
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(top_n)
            width = 0.35
            
            layer_names = [layer.split('_')[0] for layer in sorted_layers[:top_n]]
            q_times = [q_layers[layer] for layer in sorted_layers[:top_n]]
            r_times = [reg_layers[layer] for layer in sorted_layers[:top_n]]
            
            ax.bar(x - width/2, q_times, width, label='Quaternion')
            ax.bar(x + width/2, r_times, width, label='Regular')
            
            ax.set_ylabel('Time (ms)')
            ax.set_title('Layer-wise Inference Time Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(layer_names, rotation=45, ha='right')
            ax.legend()
            
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}_layer_comparison.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_results(self, output_path=None):
        """
        Export profiling results to CSV.
        
        Args:
            output_path (str, optional): Path to save CSV file
        """
        if output_path is None:
            output_path = "model_profiling_results.csv"
        
        # Prepare data for DataFrame
        data = []
        q_results = self.results['quaternion']
        
        # Quaternion model results
        for layer_name, time in q_results['layer_times'].items():
            memory = q_results['layer_memory'].get(layer_name, 0)
            data.append({
                'model_type': 'quaternion',
                'layer_name': layer_name,
                'time_ms': time,
                'memory_mb': memory
            })
        
        # Regular model results if available
        if self.regular_model:
            reg_results = self.results['regular']
            for layer_name, time in reg_results['layer_times'].items():
                memory = reg_results['layer_memory'].get(layer_name, 0)
                data.append({
                    'model_type': 'regular',
                    'layer_name': layer_name,
                    'time_ms': time,
                    'memory_mb': memory
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")
        
        # Also save summary
        summary_data = {
            'quaternion': {
                'total_time_ms': q_results['total_time'],
                'total_std_ms': q_results.get('total_std', 0),
                'params': q_results['params']
            }
        }
        
        if self.regular_model:
            reg_results = self.results['regular']
            summary_data['regular'] = {
                'total_time_ms': reg_results['total_time'],
                'total_std_ms': reg_results.get('total_std', 0),
                'params': reg_results['params']
            }
            
            comparison = self.compare_results()
            summary_data['comparison'] = {
                'speed_ratio': comparison['total_time_ratio'],
                'params_ratio': comparison['params_ratio']
            }
        
        summary_df = pd.DataFrame.from_dict(summary_data, orient='index')
        summary_output = output_path.replace('.csv', '_summary.csv')
        summary_df.to_csv(summary_output)
        print(f"Summary exported to {summary_output}")


def main():
    parser = argparse.ArgumentParser(description='Profile Quaternion and Regular YOLO models')
    parser.add_argument('--q_model', type=str, required=True, help='Path to quaternion model')
    parser.add_argument('--reg_model', type=str, default=None, help='Path to regular model for comparison')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for inference')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup runs')
    parser.add_argument('--runs', type=int, default=100, help='Number of timed runs')
    parser.add_argument('--device', type=str, default=None, help='Device to run on (cpu, 0, 1, etc.)')
    parser.add_argument('--output', type=str, default='profiling_results', help='Output filename prefix')
    args = parser.parse_args()
    
    profiler = ModelProfiler(
        q_model_path=args.q_model,
        regular_model_path=args.reg_model,
        imgsz=args.imgsz,
        num_warmup_runs=args.warmup,
        num_timed_runs=args.runs,
        device=args.device
    )
    
    results = profiler.profile()
    profiler.print_summary()
    profiler.plot_results(save_path=args.output)
    profiler.export_results(output_path=f"{args.output}.csv")


if __name__ == "__main__":
    main()


# Example 2: Compare quaternion model against a regular model
# python quaternion_profiler.py --q_model yolov8q.pt --reg_model yolov8n.pt --imgsz 640 --runs 50 --device 0