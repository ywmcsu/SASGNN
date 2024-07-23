from data_loader.forecast_dataloader import Dataset_Custom
from data_loader.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom
}

def data_provider(args, flag):
    Data = Dataset_Custom
    timeenc = 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1
        freq = 4
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = 4

    data_set = Data(
        data_path=args.dataset_path,
        flag=flag,
        size=[args.window_size, args.labellen, args.horizon],
        features='M',
        target='OT',
        timeenc=0,
        freq='h',
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last)
    return data_set, data_loader
