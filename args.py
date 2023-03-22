import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-batch_size", "--batch_size", help="batch_Size")
args = argParser.parse_args()
print("args=%s" % args)
print("args.batch_size=%s" % args.batch_size)
print("args.batch_size type=", list(args.batch_size))