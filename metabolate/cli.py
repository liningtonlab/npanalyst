from .core import *


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path',help='path to input for either task or basketed data for activity mapping')
    parser.add_argument('task',help='task to perform.',choices=['replicate','basket','both'])
    parser.add_argument('-w','--workers',help="number of parallel workers to spin up",type=int,default=0)
    parser.add_argument('-f','--filename_col',help='column name for the filename',default='Sample')
    parser.add_argument('--basket_info',help='Flag to save basket info as a json object in resulting files.',action='store_true')
    parser.add_argument('--ms2',help='flag match ms2 ions during basketing Note: Will drastically increase time required',action='store_true')
    parser.add_argument('--activity_data',help='path to activity data or folder containing multiple activity files')
    args = parser.parse_args()

    if args.task in ['replicate','both']:
        data_path = Path(args.path)
        mp_proc_folder(data_path,FILENAMECOL=args.filename_col,max_workers=args.workers,calc_basket_info=args.basket_info)

    if args.task in ['basket','both']:
        if args.task == 'both':
            data_path = data_path.joinpath('Replicated')
        else:
            data_path = Path(args.path)
        basket(data_path,args.filename_col,ms2=args.ms2,calc_basket_info=args.basket_info)

    if args.activity_data and args.task not in ('both','basket'):
        print('path argument must be path to basketed data file')
    
    if args.activity_data:
        pass
        '''do activity related funcs and make outputs'''

if __name__ == "__main__":
    main()