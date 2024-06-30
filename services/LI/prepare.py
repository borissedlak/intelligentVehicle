from demo_utils import parse_demo_configs, download_and_unzip

if __name__ == '__main__':
    configs = parse_demo_configs()

    # Try to download the dataset for demonstration
    server_url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data'
    download_url = '{}/{}/{}.zip'.format(server_url, configs.foldername[:-5], configs.foldername)
    download_and_unzip(configs.dataset_dir, download_url)

    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)
    model.eval()

    out_cap = None
    demo_dataset = Demo_KittiDataset(configs)
