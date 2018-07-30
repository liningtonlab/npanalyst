from core import *

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path',help='path to input for either task')
    parser.add_argument('task',help='task to perform.',choices=['replicate','basket','both'])
    parser.add_argument('-w','--workers',help="number of parallel workers to spin up",type=int,default=0)
    parser.add_argument('--basket_info',help='Flag to save basket info as a json object in resulting files.',action='store_true')
    args = parser.parse_args()

    if args.task in ['replicate','both']:
        data_path = Path(args.path)
        if args.workers == 1:
            proc_folder(data_path,max_workers=args.workers,calc_basket_info=args.basket_info)
        else:
            mp_proc_folder(data_path,max_workers=args.workers,calc_basket_info=args.basket_info)

    elif args.task in ['basket','both']:
        if args.task == 'both':
            data_path = 'Replicated'
        else:
            data_path = Path(args.path)
        basket(data_path)
    