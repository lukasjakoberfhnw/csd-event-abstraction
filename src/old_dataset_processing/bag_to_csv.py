import os
from bagpy import bagreader

data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
output_path = os.path.join(os.path.dirname(__file__), '..', 'output')
scand_file_location = os.path.join(data_path, 'SCAND', 'A_Jackal_Library_Fountain_Thu_Oct_28_5.bag')

def bag_to_csv(bag_file_location: str, output_file_location: str):
    bag = bagreader(bag_file_location)
    bag.topic_table.to_csv(output_file_location)

def main():
    bag_to_csv(scand_file_location, os.path.join(output_path, 'scand.csv'))

if __name__ == '__main__':
    main()