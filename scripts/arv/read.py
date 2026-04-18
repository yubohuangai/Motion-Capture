import pickle as pkl
file_path = '/Users/yubo/paper/imu/DIP_IMU_and_Others/DIP_IMU/s_01/01.pkl'
with open(file_path, 'rb') as f:
	data = pkl.load(f, encoding='latin1')
	print(data.keys())