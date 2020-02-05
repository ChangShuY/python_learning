import argparse

parser = argparse.ArgumentParser()
parser.add_argument("first")
parser.add_argument("second",type=int,help='display an integer')
args = parser.parse_args()
print(args.first+args.first,args.second*args.second)
