import hydra
import torchvision
import workspaces


@hydra.main(config_path='conf', config_name='config')
def main(cfg):
    workspace = workspaces.Workspace(cfg)


if __name__ == '__main__':
    main()
