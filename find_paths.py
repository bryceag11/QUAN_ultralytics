import os
import glob
from pathlib import Path

def find_all_msvc_installations():
    """Find all Visual Studio installations with their MSVC components"""
    program_files = os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')
    vs_path = Path(program_files) / 'Microsoft Visual Studio'
    
    if not vs_path.exists():
        print(f"❌ Could not find Visual Studio path at {vs_path}")
        return
        
    editions = ['BuildTools', 'Community', 'Professional', 'Enterprise']
    installations = []
    
    for year in ['2022', '2019', '2017']:
        for edition in editions:
            base_path = vs_path / year / edition
            if base_path.exists():
                installations.append((year, edition, base_path))
                
    return installations

def find_msvc_paths(vs_installation):
    """Find MSVC paths for a given Visual Studio installation"""
    year, edition, base_path = vs_installation
    msvc_base = base_path / 'VC' / 'Tools' / 'MSVC'
    
    if not msvc_base.exists():
        return None
        
    # Find all MSVC versions
    versions = [p for p in msvc_base.glob('*') if p.is_dir()]
    if not versions:
        return None
        
    # Get latest version
    latest_version = sorted(versions)[-1]
    
    paths = {
        'base': latest_version,
        'bin_x64': latest_version / 'bin' / 'Hostx64' / 'x64',
        'include': latest_version / 'include',
        'lib_x64': latest_version / 'lib' / 'x64',
    }
    
    return paths

def find_windows_sdk():
    """Find Windows SDK paths"""
    program_files = os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')
    sdk_root = Path(program_files) / 'Windows Kits' / '10'
    
    if not sdk_root.exists():
        print(f"❌ Could not find Windows SDK at {sdk_root}")
        return None
        
    # Find latest SDK version
    include_versions = [p for p in (sdk_root / 'Include').glob('10.*') if p.is_dir()]
    lib_versions = [p for p in (sdk_root / 'Lib').glob('10.*') if p.is_dir()]
    
    if not include_versions or not lib_versions:
        print("❌ Could not find SDK Include or Lib directories")
        return None
        
    latest_version = sorted(include_versions)[-1].name
    
    paths = {
        'root': sdk_root,
        'include': {
            'ucrt': sdk_root / 'Include' / latest_version / 'ucrt',
            'um': sdk_root / 'Include' / latest_version / 'um',
            'shared': sdk_root / 'Include' / latest_version / 'shared',
        },
        'lib': {
            'ucrt': sdk_root / 'Lib' / latest_version / 'ucrt' / 'x64',
            'um': sdk_root / 'Lib' / latest_version / 'um' / 'x64',
        },
        'version': latest_version
    }
    
    return paths

def find_cuda_installation():
    """Find CUDA installation"""
    program_files = r'C:\Program Files'
    cuda_path = Path(program_files) / 'NVIDIA GPU Computing Toolkit' / 'CUDA'
    
    if not cuda_path.exists():
        print(f"❌ Could not find CUDA at {cuda_path}")
        return None
        
    # Find all CUDA versions
    versions = [p for p in cuda_path.glob('v*') if p.is_dir()]
    if not versions:
        print("❌ No CUDA versions found")
        return None
        
    latest_version = sorted(versions)[-1]
    
    paths = {
        'root': latest_version,
        'include': latest_version / 'include',
        'lib': latest_version / 'lib' / 'x64',
        'bin': latest_version / 'bin',
        'version': latest_version.name[1:]  # Remove 'v' prefix
    }
    
    return paths

def check_path_exists(path, description):
    """Check if path exists and print status"""
    if path.exists():
        print(f"✅ Found {description}: {path}")
        return True
    else:
        print(f"❌ Missing {description}: {path}")
        return False

def main():
    print("\n=== Checking Visual Studio Installations ===")
    installations = find_all_msvc_installations()
    if not installations:
        print("❌ No Visual Studio installations found!")
        return
        
    for year, edition, path in installations:
        print(f"\nFound Visual Studio {year} {edition} at {path}")
        msvc_paths = find_msvc_paths((year, edition, path))
        if msvc_paths:
            print("\nMSVC Paths:")
            for key, path in msvc_paths.items():
                check_path_exists(path, f"MSVC {key}")
                
    print("\n=== Checking Windows SDK ===")
    sdk_paths = find_windows_sdk()
    if sdk_paths:
        print(f"\nFound Windows SDK version: {sdk_paths['version']}")
        print("\nSDK Include Paths:")
        for key, path in sdk_paths['include'].items():
            check_path_exists(path, f"SDK {key} includes")
        print("\nSDK Library Paths:")
        for key, path in sdk_paths['lib'].items():
            check_path_exists(path, f"SDK {key} libraries")
            
    print("\n=== Checking CUDA Installation ===")
    cuda_paths = find_cuda_installation()
    if cuda_paths:
        print(f"\nFound CUDA version: {cuda_paths['version']}")
        for key, path in cuda_paths.items():
            if key != 'version':
                check_path_exists(path, f"CUDA {key}")

if __name__ == "__main__":
    main()