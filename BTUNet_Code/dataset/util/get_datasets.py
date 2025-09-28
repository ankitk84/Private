def get_datasets(dataset_name, data_root, train, transforms=None):
    if dataset_name == 'DRIVE':
        from ..DRIVE_dataset import DRIVE_dataset
        dataset = DRIVE_dataset(data_root=data_root, train=train, transforms=transforms)
        num_return = dataset.num_return
        
    elif dataset_name == 'STARE':
        from ..STARE_dataset import STARE_dataset
        dataset = STARE_dataset(data_root=data_root, train=train, transforms=transforms)
        num_return = dataset.num_return
        
    elif dataset_name == 'CHASEDB':
        from ..CHASEDB_dataset import CHASEDB_dataset
        dataset = CHASEDB_dataset(data_root=data_root, train=train, transforms=transforms)
        num_return = dataset.num_return
        
    elif dataset_name == 'HRF':
        from ..HRF_dataset import HRF_dataset
        dataset = HRF_dataset(data_root=data_root, train=train, transforms=transforms)
        num_return = dataset.num_return
    
    elif dataset_name == 'DRIVEAUG':    
        from ..ALL_dataset import ALL_Dataset
        dataset = ALL_Dataset(data_root=data_root, train=train, transforms=transforms)
        num_return = dataset.num_return

    elif dataset_name == 'IOSTAR':    
        from ..IOSTAR_dataset import IOSTAR_dataset
        dataset = IOSTAR_dataset(data_root=data_root, train=train, transforms=transforms)
        num_return = dataset.num_return
        
    else:
        raise NotImplementedError
        
        
    return dataset, num_return
