import torch
from torch.utils.data import Dataset

from .mot_sequence import MOT17_Sequence, MOT19CVPR_Sequence, MOT17LOWFPS_Sequence,MOTS17_Sequence


class MOT17_Wrapper(Dataset):
	"""A Wrapper for the MOT_Sequence class to return multiple sequences."""

	def __init__(self, split, dets, dataloader):
		"""Initliazes all subset of the dataset.

		Keyword arguments:
		split -- the split of the dataset to use
		dataloader -- args for the MOT_Sequence dataloader
		"""
		train_sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
		test_sequences = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14']



		if "train" == split:
			sequences = train_sequences
		elif "test" == split:
			sequences = test_sequences
		elif "all" == split:
			sequences = train_sequences + test_sequences
		elif f"MOT17-{split}" in train_sequences + test_sequences:
			sequences = [f"MOT17-{split}"]
		# elif "train_mask" == split:
			# sequences_mask = train_mask_sequences
			# sequences = train_det_4_mask_sequences
		else:
			raise NotImplementedError("MOT split not available.")

		self._data = []
		for s in sequences:
			if dets == '17':
				self._data.append(MOT17_Sequence(seq_name=s, dets='DPM17', **dataloader))
				self._data.append(MOT17_Sequence(seq_name=s, dets='FRCNN17', **dataloader))
				self._data.append(MOT17_Sequence(seq_name=s, dets='SDP17', **dataloader))
			# elif dets == 'mots17':
			# 	self._data.append(MOT17_Sequence(seq_name=s, dets='FRCNN17', **dataloader))
			else:
				self._data.append(MOT17_Sequence(seq_name=s, dets=dets, **dataloader))
		# if sequences_mask:
		# 	self._data.append(MOT17_Sequence(Seq_name="mots",dets='FRCNN17',**dataloader))

	def __len__(self):
		return len(self._data)

	def __getitem__(self, idx):
		return self._data[idx]


class MOTS17_Wrapper(MOT17_Wrapper):
	"""A Wrapper for the MOTS_Sequence class to return multiple sequences."""

	def __init__(self, split, dataloader):
		"""Initliazes all subset of the dataset.

		Keyword arguments:
		split -- the split of the dataset to use
		dataloader -- args for the MOT_Sequence dataloader
		"""

		train_det_for_mask_sequences = ['MOT17-%02d'%idx for idx in [2,5,9,11]]#mots challenge provided the mask anno data.
		train_mask_sequences =  ["%04d"%idx for idx in [2,5,9,11]]   #这是为了训练
		train_sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']#这是为了测试　训练集
		test_sequences = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14']

		if "train_mask" == split:
			sequences = train_det_for_mask_sequences
			# mask_sequences = train_mask_sequences
		elif "train"==split:
			sequences = train_sequences
		elif "test" == split:
			sequences = test_sequences
		elif "all" == split:
			sequences = train_sequences + test_sequences
		elif f"MOT17-{split}" in test_sequences:#单独测试一个
			sequences = [f"MOT17-{split}"]
			# mask_sequences = ["%04d"%split]
		else:
			raise NotImplementedError("MOT19CVPR split not available.")

		self._data = []
		for s in sequences:
			self._data.append(MOTS17_Sequence(seq_name=s, **dataloader))
		# for s in mask_sequences:
		# 	self._data.append(MOTS17_Sequence(seq_name=s, **dataloader))
	def __len__(self):
		return len(self._data)

	def __getitem__(self, idx):
		return self._data[idx]


class MOT19CVPR_Wrapper(MOT17_Wrapper):
	"""A Wrapper for the MOT_Sequence class to return multiple sequences."""

	def __init__(self, split, dataloader):
		"""Initliazes all subset of the dataset.

		Keyword arguments:
		split -- the split of the dataset to use
		dataloader -- args for the MOT_Sequence dataloader
		"""
		train_sequences = ['CVPR19-01', 'CVPR19-02', 'CVPR19-03', 'CVPR19-05']
		test_sequences = ['CVPR19-04', 'CVPR19-06', 'CVPR19-07', 'CVPR19-08']

		if "train" == split:
			sequences = train_sequences
		elif "test" == split:
			sequences = test_sequences
		elif "all" == split:
			sequences = train_sequences + test_sequences
		elif f"CVPR19-{split}" in train_sequences + test_sequences:
			sequences = [f"CVPR19-{split}"]
		else:
			raise NotImplementedError("MOT19CVPR split not available.")

		self._data = []
		for s in sequences:
			self._data.append(MOT19CVPR_Sequence(seq_name=s, **dataloader))

	def __len__(self):
		return len(self._data)

	def __getitem__(self, idx):
		return self._data[idx]


class MOT17LOWFPS_Wrapper(MOT17_Wrapper):
	"""A Wrapper for the MOT_Sequence class to return multiple sequences."""

	def __init__(self, split, dataloader):
		"""Initliazes all subset of the dataset.

		Keyword arguments:
		split -- the split of the dataset to use
		dataloader -- args for the MOT_Sequence dataloader
		"""

		sequences = ['MOT17-02', 'MOT17-04', 'MOT17-09', 'MOT17-10', 'MOT17-11']

		self._data = []
		for s in sequences:
			self._data.append(MOT17LOWFPS_Sequence(split=split, seq_name=s, dets='FRCNN17', **dataloader))

	def __len__(self):
		return len(self._data)

	def __getitem__(self, idx):
		return self._data[idx]
