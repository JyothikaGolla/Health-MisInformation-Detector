import os, sys, platform, subprocess

def main():
    print("Installing PyTorch Geometric (CPU by default). For CUDA, edit the URLs per your CUDA version.")
    # Adjust wheels per https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
    cmds = [
        [sys.executable, '-m', 'pip', 'install', 'torch-scatter', 'torch-sparse', 'torch-cluster', 'torch-spline-conv', '-f', 'https://data.pyg.org/whl/torch-2.3.0+cpu.html'],
        [sys.executable, '-m', 'pip', 'install', 'torch-geometric']
    ]
    for c in cmds:
        print('>', ' '.join(c))
        subprocess.check_call(c)

if __name__ == '__main__':
    main()
