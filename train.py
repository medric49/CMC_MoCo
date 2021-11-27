from pathlib import Path

import hydra
import torchvision
import workspaces


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    workspace = workspaces.Workspace(cfg)
    root_dir = Path.cwd()
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
